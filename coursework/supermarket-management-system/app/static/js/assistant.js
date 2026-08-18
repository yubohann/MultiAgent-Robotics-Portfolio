(() => {
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-message');
    const clearChatBtn = document.getElementById('clear-chat');
    const charCount = document.getElementById('char-count');
    const quickActionBtns = document.querySelectorAll('.quick-action-btn');
    const faqBtns = document.querySelectorAll('.mt-6 .space-y-2 button');

    if (!chatMessages || !chatInput || !sendButton || !clearChatBtn || !charCount) {
        return;
    }

    let isProcessing = false;

    chatInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = `${Math.min(this.scrollHeight, 128)}px`;

        const count = this.value.length;
        charCount.textContent = count;

        sendButton.disabled = count === 0 || count > 500 || isProcessing;

        if (count > 500) {
            charCount.classList.add('text-red-500');
        } else {
            charCount.classList.remove('text-red-500');
        }
    });

    const sendMessage = async () => {
        const message = chatInput.value.trim();
        if (!message || message.length > 500 || isProcessing) {
            return;
        }

        addUserMessage(message);

        chatInput.value = '';
        chatInput.style.height = 'auto';
        charCount.textContent = '0';
        sendButton.disabled = true;

        showTypingIndicator();

        try {
            const response = await fetch('/api/assistant/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });

            const result = await response.json();
            removeTypingIndicator();

            if (!response.ok || !result.success) {
                addBotMessage(result.answer || '抱歉，暂时无法处理您的请求，请稍后重试。');
                return;
            }

            let reply = result.answer || '已收到您的问题。';
            if (Array.isArray(result.suggestions) && result.suggestions.length > 0) {
                reply += `\n\n您还可以问：\n${result.suggestions.slice(0, 3).map((item) => `- ${item}`).join('\n')}`;
            }
            addBotMessage(reply);
        } catch (error) {
            removeTypingIndicator();
            addBotMessage('网络连接异常，请检查后重试。');
        }
    };

    const sendMessageWithText = (text) => {
        if (isProcessing) {
            return;
        }
        chatInput.value = text;
        chatInput.dispatchEvent(new Event('input'));
        sendMessage();
    };

    const addUserMessage = (message) => {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message flex items-start space-x-3 justify-end';
        messageDiv.innerHTML = `
            <div class="flex-1 flex flex-col items-end">
                <div class="bg-primary text-white p-4 rounded-2xl rounded-tr-none shadow-md max-w-[85%]">
                    <p class="text-sm leading-relaxed">${escapeChatHtml(message)}</p>
                </div>
                <p class="text-xs text-slate-400 mt-1 mr-1">${getCurrentTime()}</p>
            </div>
            <div class="w-10 h-10 bg-slate-200 rounded-full flex items-center justify-center flex-shrink-0">
                <span class="text-sm font-bold text-slate-600">我</span>
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    };

    const addBotMessage = (message) => {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message flex items-start space-x-3';
        messageDiv.innerHTML = `
            <div class="w-10 h-10 bg-gradient-to-r from-primary to-secondary rounded-full flex items-center justify-center flex-shrink-0 shadow-md">
                <div class="text-white text-xs font-bold">AI</div>
            </div>
            <div class="flex-1">
                <div class="bg-white p-4 rounded-2xl rounded-tl-none shadow-sm border border-slate-100 max-w-[85%]">
                    <p class="text-sm text-slate-700 leading-relaxed whitespace-pre-line">${escapeChatHtml(message)}</p>
                </div>
                <p class="text-xs text-slate-400 mt-1 ml-1">${getCurrentTime()}</p>
            </div>
        `;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    };

    const showTypingIndicator = () => {
        isProcessing = true;
        sendButton.disabled = true;

        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator';
        typingDiv.className = 'chat-message flex items-start space-x-3';
        typingDiv.innerHTML = `
            <div class="w-10 h-10 bg-gradient-to-r from-primary to-secondary rounded-full flex items-center justify-center flex-shrink-0 shadow-md">
                <div class="text-white text-xs font-bold">AI</div>
            </div>
            <div class="bg-white p-4 rounded-2xl rounded-tl-none shadow-sm border border-slate-100">
                <div class="typing-indicator flex space-x-1">
                    <span class="w-2 h-2 bg-slate-400 rounded-full"></span>
                    <span class="w-2 h-2 bg-slate-400 rounded-full"></span>
                    <span class="w-2 h-2 bg-slate-400 rounded-full"></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        scrollToBottom();
    };

    const removeTypingIndicator = () => {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        isProcessing = false;
        sendButton.disabled = chatInput.value.trim().length === 0;
    };

    const scrollToBottom = () => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    const getCurrentTime = () => {
        const now = new Date();
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        return `${hours}:${minutes}`;
    };

    const escapeChatHtml = (text) => {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    };

    clearChatBtn.addEventListener('click', () => {
        if (confirm('确定要清空所有对话记录吗？')) {
            chatMessages.innerHTML = '';
            addBotMessage('对话已清空。有什么我可以帮您的吗？');
        }
    });

    sendButton.addEventListener('click', sendMessage);

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    quickActionBtns.forEach((btn) => {
        btn.addEventListener('click', function () {
            const actionText = this.querySelector('p.font-medium').textContent;
            sendMessageWithText(`我想${actionText}`);
        });
    });

    faqBtns.forEach((btn) => {
        btn.addEventListener('click', function () {
            const question = this.textContent.trim();
            sendMessageWithText(question);
        });
    });

    scrollToBottom();
})();
