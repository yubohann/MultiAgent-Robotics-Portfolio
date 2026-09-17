"""Original SplitGNN training script driven by YAML configs and preprocessed graphs."""

import os
import time
import numpy as np
import torch
import dgl
import dgl.nn as dglnn
import torch.optim as optim
import torch.nn.functional as F

try:
    from .model import *
    from .utils import *
except ImportError:
    from model import *
    from utils import *


import warnings
warnings.filterwarnings('ignore')
random.seed(42)

if __name__ == '__main__':
    # Parse CLI arguments and set the random seed.
    args = parse_args()
    setup_seed(args.seed)
    requested_device = str(args.cuda).lower()
    if requested_device != 'cpu' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda}')
    else:
        device = torch.device('cpu')
    args.device = device
    dataset_path = resolve_graph_path(args.data_path, args.dataset)
    model_path = os.path.join(args.result_path, args.dataset+'_'+str(args.gamma)+'_model.pt')
    gmodel_path = os.path.join(args.result_path, args.dataset+'_'+str(args.gamma)+'_gmodel.pt')
    results = {'F1-macro':[],'AUC':[],'G-Mean':[],'recall':[]}
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    writer = None
    if getattr(args, 'tb', False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            run_name = args.run_name.strip() if getattr(args, 'run_name', '') else ''
            if not run_name:
                run_name = f'{args.dataset}_g{args.gamma}_seed{args.seed}_{time.strftime("%Y%m%d-%H%M%S")}'
            tb_run_dir = os.path.join(args.tb_dir, run_name)
            os.makedirs(tb_run_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_run_dir)
            writer.add_text('hparams', str(vars(args)))
            print(f'TensorBoard enabled. Log dir: {tb_run_dir}')
        except Exception as e:
            print(f'Failed to enable TensorBoard logging: {e}')
            writer = None
    # Load the dataset and normalize node features to a stable scale.
    dataset = dgl.load_graphs(dataset_path)[0][0]
    features = dataset.ndata['feature'].numpy()
    if args.dataset == 'amazon':
        features = np.delete(features, 19, axis=1) # remove label leakage feature
    features = normalize(features)
    features = torch.from_numpy(features).float()
    dataset.ndata['feature'] = features
    dataset = dataset.to(device)
    
    # Start training.
    print('Start training model...')
    model = SplitGNN(args, dataset)
    model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stop = EarlyStop(args.early_stop)
    gearly_stop = EarlyStop(args.early_stop)
    best_auc_threshold = 0.5
    best_gmean_threshold = 0.5
    last_epoch = -1
    for e in range(args.epoch):
        last_epoch = e
        
        model.train()
        loss = model.loss(dataset) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if writer is not None:
            writer.add_scalar('train/loss', float(loss.item()), e)
        
        with torch.no_grad():
            # Score on the validation split and save checkpoints by validation AUC and G-Mean.
            model.eval()
            valid_mask = dataset.ndata['valid_mask'].bool()
            valid_labels = dataset.ndata['label'][valid_mask].cpu().numpy()
            valid_logits = model(dataset)[valid_mask]
            valid_metrics = evaluate(valid_labels, valid_logits, threshold=None, return_details=True)
            f1_macro = valid_metrics['f1_macro']
            auc = valid_metrics['auc']
            gmean = valid_metrics['gmean']
            recall = valid_metrics['recall']
            if writer is not None:
                writer.add_scalar('valid/auc', float(auc), e)
                writer.add_scalar('valid/gmean', float(gmean), e)
                writer.add_scalar('valid/f1_macro', float(f1_macro), e)
                writer.add_scalar('valid/recall', float(recall), e)
                writer.add_scalar('valid/threshold', float(valid_metrics['threshold']), e)
                writer.add_scalar('valid/positive_rate', float(valid_metrics['positive_rate']), e)
                writer.add_scalar('valid/prob_std', float(valid_metrics['prob_std']), e)
                writer.add_scalar('valid/best_auc', float(early_stop.best_eval), e)
                writer.add_scalar('valid/best_gmean', float(gearly_stop.best_eval), e)
            
            if e % 10 == 0 and args.log:
                print(f'{e}: Best Epoch:{early_stop.best_epoch}, Best valid AUC:{early_stop.best_eval}, Loss:{loss.item()}, Current valid: Recall:{recall}, F1_macro:{f1_macro}, G-Mean:{gmean}, AUC:{auc}')
            do_store, do_stop = early_stop.step(auc, e)
            gmean_store, gmean_stop = gearly_stop.step(gmean, e)
            if do_store:
                torch.save(model, model_path)
                best_auc_threshold = float(valid_metrics['threshold'])
                if writer is not None:
                    writer.add_scalar('checkpoint/saved_best_auc_epoch', float(e), e)
            if gmean_store:
                torch.save(model, gmodel_path)
                best_gmean_threshold = float(valid_metrics['threshold'])
                if writer is not None:
                    writer.add_scalar('checkpoint/saved_best_gmean_epoch', float(e), e)
            if do_stop:
                break
    print('End training')
    # Run one test pass with the best validation AUC model.
    print('Test model...')
    model = torch.load(model_path, map_location=device)
    with torch.no_grad():
        model.eval()
        test_mask = dataset.ndata['test_mask'].bool()
        test_labels = dataset.ndata['label'][test_mask]
        test_labels = test_labels.cpu().numpy()
        logits = model(dataset)[test_mask]
        logits = logits.cpu()
        test_result_path = os.path.join(args.result_path, args.dataset+'_'+str(args.gamma))
        test_metrics = evaluate(test_labels, logits, test_result_path, threshold=best_auc_threshold, return_details=True)
        f1_macro = test_metrics['f1_macro']
        auc = test_metrics['auc']
        gmean = test_metrics['gmean']
        recall = test_metrics['recall']
        results['F1-macro'].append(f1_macro)
        results['AUC'].append(auc)
        results['G-Mean'].append(gmean)
        results['recall'].append(recall)
        print(f'Test: Recall:{recall}, F1-macro:{f1_macro}, AUC:{auc}, G-Mean:{gmean}')
        if writer is not None:
            test_step = last_epoch + 1
            writer.add_scalar('test_best_auc/auc', float(auc), test_step)
            writer.add_scalar('test_best_auc/gmean', float(gmean), test_step)
            writer.add_scalar('test_best_auc/f1_macro', float(f1_macro), test_step)
            writer.add_scalar('test_best_auc/recall', float(recall), test_step)
            writer.add_scalar('test_best_auc/threshold', float(test_metrics['threshold']), test_step)
        
    # Run one test pass with the best validation G-Mean model.
    model = torch.load(gmodel_path, map_location=device)
    with torch.no_grad():
        model.eval()
        test_mask = dataset.ndata['test_mask'].bool()
        test_labels = dataset.ndata['label'][test_mask]
        test_labels = test_labels.cpu().numpy()
        logits = model(dataset)[test_mask]
        logits = logits.cpu()
        test_result_path = os.path.join(args.result_path, args.dataset+'_'+str(args.gamma)+'g')
        test_metrics = evaluate(test_labels, logits, test_result_path, threshold=best_gmean_threshold, return_details=True)
        f1_macro = test_metrics['f1_macro']
        auc = test_metrics['auc']
        gmean = test_metrics['gmean']
        recall = test_metrics['recall']
        results['F1-macro'].append(f1_macro)
        results['AUC'].append(auc)
        results['G-Mean'].append(gmean)
        results['recall'].append(recall)
        print(f'Test: Recall:{recall}, F1-macro:{f1_macro}, AUC:{auc}, G-Mean:{gmean}')
        if writer is not None:
            test_step = last_epoch + 1
            writer.add_scalar('test_best_gmean/auc', float(auc), test_step)
            writer.add_scalar('test_best_gmean/gmean', float(gmean), test_step)
            writer.add_scalar('test_best_gmean/f1_macro', float(f1_macro), test_step)
            writer.add_scalar('test_best_gmean/recall', float(recall), test_step)
            writer.add_scalar('test_best_gmean/threshold', float(test_metrics['threshold']), test_step)

    if writer is not None:
        writer.flush()
        writer.close()
    
    
