# Run Strictly By instruction.md

This guide follows `instruction.md` section order. The original document uses placeholder folders `python/` and `code/flink-ai-project/`; in this portfolio they correspond to:

- `python/` -> `recommender-services/`
- `code/flink-ai-project/` -> `flink-jobs/realtime-recommender-flink-job/`

Use a WSL terminal, not PowerShell.

## 0. Enter Project And Check Infrastructure

```bash
cd /path/to/stream-batch-lakehouse-ai-portfolio/03-ai-recommender
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

Screenshot if needed: Kafka, Flink, Spark, MinIO containers are running.

## 1. Experiment 1 - Kafka Topics And Data Generator

Create the four topics from `instruction.md`:

```bash
cd recommender-services
bash create_instruction_topics.sh | tee topics_created.log
```

Expected topics:

```text
user_behaviors
fast_recommendations
deep_recommendations
final_recommendations
```

Start the user behavior producer. Use `python3.10` in this environment because `python3` may point to another Python version without the required packages.

```bash
timeout 20s python3.10 -u user_behavior_producer.py | tee user_behavior_producer_instruction.log
```

Screenshot 03-1:

- four Kafka topics
- user behavior producer output
- behavior records contain `user_id`, `product_id`, `category`, behavior/action fields

## 2. Experiment 2 - Flink Fast Recommendation

Build the Flink fast recommender:

```bash
cd ../flink-jobs/realtime-recommender-flink-job
mvn -DskipTests clean package | tee ../../recommender-services/flink_fast_build.log
cd ../../recommender-services
```

For screenshot evidence, capture:

- `Building realtime-recommender-flink-job`
- `BUILD SUCCESS`
- `FastRecommender.java` exists in the job source tree

Run a short fast recommendation verification with Kafka:

Terminal A:

```bash
cd /path/to/stream-batch-lakehouse-ai-portfolio/03-ai-recommender/recommender-services
timeout 45s docker exec bigdata-kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic fast_recommendations \
  --from-beginning \
  --max-messages 5 | tee fast_recommendations_messages.jsonl
```

Terminal B:

```bash
cd /path/to/stream-batch-lakehouse-ai-portfolio/03-ai-recommender/flink-jobs/realtime-recommender-flink-job
timeout 45s java -cp target/realtime-recommender-flink-job-1.0-SNAPSHOT.jar com.edu.bigdata.FastRecommender | tee ../../recommender-services/flink_fast_run.log
```

Terminal C, while Terminal B is running:

```bash
cd /path/to/stream-batch-lakehouse-ai-portfolio/03-ai-recommender/recommender-services
timeout 30s python3.10 -u user_behavior_producer.py | tee user_behavior_for_flink.log
```

Screenshot 03-2:

- `BUILD SUCCESS`
- fast recommendation output or `fast_recommendations_messages.jsonl`
- output contains `user_id`, `trigger_product`, `recommendations`, `recommendation_type: fast`

## 3. Experiment 3 - DeepFM Training

Install/check dependencies:

```bash
python3.10 test_dependencies.py | tee test_dependencies.log
```

Train DeepFM:

```bash
python3.10 deepfm_recommender.py | tee deepfm_training.log
```

Screenshot 03-3:

- dependency check includes `torch`, `pandas`, `numpy`
- DeepFM training output contains `Epoch 1/5` through `Epoch 5/5`
- model saved to `deepfm_model.pth`
- metadata saved to `metadata.pkl`
- recommendation examples for `USER_00001`, `USER_00002`, `USER_00003`

## 4. Experiment 4 - Recommendation Fusion

Send several DeepFM recommendation messages to the `deep_recommendations` topic:

```bash
python3.10 send_deep_recommendations.py | tee deep_recommendations_sent.log
```

Start the fusion service:

```bash
timeout 90s python3.10 recommendation_fusion.py | tee recommendation_fusion_run.log
```

While the fusion service is running, send fast recommendation messages by running the Flink/producer pair from Experiment 2. The producer alone only writes `user_behaviors`; Flink is what turns those events into `fast_recommendations`.

Monitor final recommendations:

```bash
timeout 90s docker exec bigdata-kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic final_recommendations \
  --from-beginning \
  --max-messages 5 | tee final_recommendations_messages.jsonl
```

Screenshot 03-4:

- `recommendation_fusion.py` service output
- weights `fast` and `deep`
- `final_recommendations` messages
- messages contain `user_id`, `recommendations`, `fusion_weights`, `source: hybrid_fusion`

## 5. Experiment 5 - Complete System Verification

The Python-version full system from `instruction.md` is:

Terminal 1:

```bash
cd /path/to/stream-batch-lakehouse-ai-portfolio/03-ai-recommender/recommender-services
python3.10 -u user_behavior_producer.py
```

Terminal 2:

```bash
cd /path/to/stream-batch-lakehouse-ai-portfolio/03-ai-recommender/recommender-services
python3.10 -u realtime_recommendation_service.py
```

Terminal 3:

```bash
cd /path/to/stream-batch-lakehouse-ai-portfolio/03-ai-recommender/recommender-services
python3.10 -u advanced_recommendation_consumer.py
```

For a bounded, screenshot-friendly run:

```bash
bash run_e2e_smoke_test.sh | tee ai_recommender_verification.log
```

Screenshot 03-5:

- `consumer exit code: 0`
- recommendation messages contain `user_id`, `trigger_product`, `recommendations`
- `recommendation message count` is greater than 0

## Screenshot Checklist

- 03-1: Kafka topics + user behavior producer
- 03-2: Flink fast recommender build + fast recommendation messages
- 03-3: DeepFM dependency check + training + model saved
- 03-4: fusion service + final recommendations
- 03-5: complete Python real-time system smoke test
