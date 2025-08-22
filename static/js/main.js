const chatLog = document.getElementById('chat-log');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const submitButton = document.getElementById('submit-button');
const sendIcon = document.getElementById('send-icon');
const loadingSpinner = document.getElementById('loading-spinner');
const newChatButton = document.getElementById('new-chat-button');

const resourceTabs = document.getElementById('resource-tabs');

const promptModalOverlay = document.getElementById('prompt-modal-overlay');
const promptModalContent = document.getElementById('prompt-modal-content');
const promptModalForm = document.getElementById('prompt-modal-form');
const promptModalTitle = document.getElementById('prompt-modal-title');
const promptModalInputs = document.getElementById('prompt-modal-inputs');
const promptModalClose = document.getElementById('prompt-modal-close');

const mainContent = document.getElementById('main-content');

const statusWindow = document.getElementById('status-window');
const statusWindowContent = document.getElementById('status-window-content');
const toggleStatusButton = document.getElementById('toggle-status-button');
const statusCollapseIcon = document.getElementById('status-collapse-icon');
const statusExpandIcon = document.getElementById('status-expand-icon');
const toggleStatusCheckbox = document.getElementById('toggle-status-checkbox');

const sessionHistoryPanel = document.getElementById('session-history-panel');
const sessionList = document.getElementById('session-list');
const toggleHistoryButton = document.getElementById('toggle-history-button');
const historyCollapseIcon = document.getElementById('history-collapse-icon');
const historyExpandIcon = document.getElementById('history-expand-icon');
const toggleHistoryCheckbox = document.getElementById('toggle-history-checkbox');

const toolHeader = document.getElementById('tool-header');
const toggleHeaderButton = document.getElementById('toggle-header-button');
const headerCollapseIcon = document.getElementById('header-collapse-icon');
const headerExpandIcon = document.getElementById('header-expand-icon');
const toggleHeaderCheckbox = document.getElementById('toggle-header-checkbox');

const infoButton = document.getElementById('info-button');
const infoModalOverlay = document.getElementById('info-modal-overlay');
const infoModalClose = document.getElementById('info-modal-close');
const infoModalContent = document.getElementById('info-modal-content');

const configMenuButton = document.getElementById('config-menu-button');
const configModalOverlay = document.getElementById('config-modal-overlay');
const configModalContent = document.getElementById('config-modal-content');
const configModalClose = document.getElementById('config-modal-close');
const configForm = document.getElementById('config-form');
const configStatus = document.getElementById('config-status');
const configLoadingSpinner = document.getElementById('config-loading-spinner');
const configActionButton = document.getElementById('config-action-button');
const configActionButtonText = document.getElementById('config-action-button-text');

const refreshModelsButton = document.getElementById('refresh-models-button');
const refreshIcon = document.getElementById('refresh-icon');
const refreshSpinner = document.getElementById('refresh-spinner');
const llmProviderSelect = document.getElementById('llm-provider');
const llmModelSelect = document.getElementById('llm-model');
const llmApiKeyInput = document.getElementById('llm-api-key');
const apiKeyContainer = document.getElementById('api-key-container');
const awsCredentialsContainer = document.getElementById('aws-credentials-container');
const awsAccessKeyIdInput = document.getElementById('aws-access-key-id');
const awsSecretAccessKeyInput = document.getElementById('aws-secret-access-key');
const awsRegionInput = document.getElementById('aws-region');
const awsListingMethodContainer = document.getElementById('aws-listing-method-container');
const ollamaHostContainer = document.getElementById('ollama-host-container');
const ollamaHostInput = document.getElementById('ollama-host');

const chartingIntensitySelect = document.getElementById('charting-intensity');

const teradataStatusDot = document.getElementById('teradata-status-dot');
const llmStatusDot = document.getElementById('llm-status-dot');
const contextStatusDot = document.getElementById('context-status-dot');

const windowMenuButton = document.getElementById('window-menu-button');
const windowDropdownMenu = document.getElementById('window-dropdown-menu');

const promptEditorButton = document.getElementById('prompt-editor-button');
const promptEditorOverlay = document.getElementById('prompt-editor-overlay');
const promptEditorContent = document.getElementById('prompt-editor-content');
const promptEditorTitle = document.getElementById('prompt-editor-title');
const promptEditorTextarea = document.getElementById('prompt-editor-textarea');
const promptEditorSave = document.getElementById('prompt-editor-save');
const promptEditorReset = document.getElementById('prompt-editor-reset');
const promptEditorClose = document.getElementById('prompt-editor-close');
const promptEditorStatus = document.getElementById('prompt-editor-status');

const confirmModalOverlay = document.getElementById('confirm-modal-overlay');
const confirmModalContent = document.getElementById('confirm-modal-content');
const confirmModalTitle = document.getElementById('confirm-modal-title');
const confirmModalBody = document.getElementById('confirm-modal-body');
const confirmModalConfirm = document.getElementById('confirm-modal-confirm');
const confirmModalCancel = document.getElementById('confirm-modal-cancel');

const systemPromptPopupOverlay = document.getElementById('system-prompt-popup-overlay');
const systemPromptPopupContent = document.getElementById('system-prompt-popup-content');
const systemPromptPopupTitle = document.getElementById('system-prompt-popup-title');
const systemPromptPopupText = document.getElementById('system-prompt-popup-text');
const systemPromptPopupClose = document.getElementById('system-prompt-popup-close');
const disabledCapabilitiesContainer = document.getElementById('disabled-capabilities-container');

const chatModalButton = document.getElementById('chat-modal-button');
const chatModalOverlay = document.getElementById('chat-modal-overlay');
const chatModalContent = document.getElementById('chat-modal-content');
const chatModalClose = document.getElementById('chat-modal-close');
const chatModalForm = document.getElementById('chat-modal-form');
const chatModalInput = document.getElementById('chat-modal-input');
const chatModalLog = document.getElementById('chat-modal-log');

const viewPromptModalOverlay = document.getElementById('view-prompt-modal-overlay');
const viewPromptModalContent = document.getElementById('view-prompt-modal-content');
const viewPromptModalTitle = document.getElementById('view-prompt-modal-title');
const viewPromptModalText = document.getElementById('view-prompt-modal-text');
const viewPromptModalClose = document.getElementById('view-prompt-modal-close');

let simpleChatHistory = [];
let currentProvider = 'Google';
let currentModel = '';
let currentStatusId = 0;
let currentSessionId = null;
let resourceData = { tools: {}, prompts: {}, resources: {}, charts: {} };
let currentlySelectedResource = null;
let eventSource = null;
let systemPromptPopupTimer = null;
let countdownValue = 5;
let mouseMoveHandler = null;
let pristineConfig = {};
let isMouseOverStatus = false;
let isInFastPath = false;

// --- MODIFICATION START: Add timeouts for all indicators ---
let mcpIndicatorTimeout = null;
let llmIndicatorTimeout = null;
let contextIndicatorTimeout = null;
// --- MODIFICATION END ---

let defaultPromptsCache = {};

const thinkingIndicator = document.getElementById('thinking-indicator');
const promptNameDisplay = document.getElementById('prompt-name-display');

statusWindowContent.addEventListener('mouseenter', () => { isMouseOverStatus = true; });
statusWindowContent.addEventListener('mouseleave', () => { isMouseOverStatus = false; });

function getSystemPrompts() {
    try {
        const prompts = localStorage.getItem('userSystemPrompts');
        return prompts ? JSON.parse(prompts) : {};
    } catch (e) {
        console.error("Could not parse system prompts from localStorage", e);
        return {};
    }
}

function getNormalizedModelId(modelId) {
    if (!modelId) return '';
    if (modelId.startsWith('arn:aws:bedrock:')) {
        const parts = modelId.split('/');
        const modelPart = parts[parts.length - 1];
        return modelPart.replace(/^(eu|us|apac)\./, '');
    }
    return modelId;
}

function getPromptStorageKey(provider, model) {
    const normalizedModel = getNormalizedModelId(model);
    return `${provider}-${normalizedModel}`;
}

function saveSystemPromptForModel(provider, model, promptText, isCustom) {
    const prompts = getSystemPrompts();
    const key = getPromptStorageKey(provider, model);
    prompts[key] = { prompt: promptText, isCustom: isCustom };
    localStorage.setItem('userSystemPrompts', JSON.stringify(prompts));
}

function getSystemPromptForModel(provider, model) {
    const prompts = getSystemPrompts();
    const key = getPromptStorageKey(provider, model);
    return prompts[key]?.prompt || null;
}

function isPromptCustomForModel(provider, model) {
    const prompts = getSystemPrompts();
    const key = getPromptStorageKey(provider, model);
    return prompts[key]?.isCustom || false;
}

async function getDefaultSystemPrompt(provider, model) {
    const key = getPromptStorageKey(provider, model);
    if (defaultPromptsCache[key]) {
        return defaultPromptsCache[key];
    }

    try {
        const res = await fetch(`/system_prompt/${provider}/${getNormalizedModelId(model)}`);
        if (!res.ok) {
            throw new Error(`Failed to fetch default prompt: ${res.statusText}`);
        }
        const data = await res.json();
        if (data.system_prompt) {
            defaultPromptsCache[key] = data.system_prompt;
            return data.system_prompt;
        }
        throw new Error("Server response did not contain a system_prompt.");
    } catch (e) {
        console.error(`Error getting default system prompt for ${key}:`, e);
        promptEditorStatus.textContent = 'Error fetching default prompt.';
        promptEditorStatus.className = 'text-sm text-red-400';
        return null;
    }
}


function copyToClipboard(button) {
    const codeBlock = button.closest('.sql-code-block').querySelector('code');
    const textToCopy = codeBlock.innerText;

    navigator.clipboard.writeText(textToCopy).then(() => {
        const originalContent = button.innerHTML;
        button.textContent = 'Copied!';
        button.classList.add('copied');
        setTimeout(() => {
            button.innerHTML = originalContent;
            button.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}

function copyTableToClipboard(button) {
    const dataStr = button.dataset.table;
    if (!dataStr) {
        console.error("No data-table attribute found on the button.");
        return;
    }

    try {
        const data = JSON.parse(dataStr);
        if (!Array.isArray(data) || data.length === 0) {
            return;
        }

        const headers = Object.keys(data[0]);
        let tsvContent = headers.join('\t') + '\n';

        data.forEach(row => {
            const values = headers.map(header => {
                let value = row[header] === null || row[header] === undefined ? '' : String(row[header]);
                value = value.replace(/\t/g, ' ').replace(/\n/g, ' ');
                return value;
            });
            tsvContent += values.join('\t') + '\n';
        });

        navigator.clipboard.writeText(tsvContent).then(() => {
            const originalContent = button.innerHTML;
            button.textContent = 'Copied!';
            button.classList.add('copied');
            setTimeout(() => {
                button.innerHTML = originalContent;
                button.classList.remove('copied');
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy table data: ', err);
        });

    } catch (e) {
        console.error("Failed to parse or process table data for copying:", e);
    }
}

function renderChart(containerId, spec) {
    try {
        const chartSpec = typeof spec === 'string' ? JSON.parse(spec) : spec;
        const plot = new G2Plot[chartSpec.type](containerId, chartSpec.options);
        plot.render();
    } catch (e) {
        console.error("Failed to render chart:", e);
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div class="p-4 text-red-400">Error rendering chart: ${e.message}</div>`;
        }
    }
}

function addMessage(role, content) {
    const wrapper = document.createElement('div');
    wrapper.className = `message-bubble flex items-start gap-4 ${role === 'user' ? 'justify-end' : ''}`;
    const icon = document.createElement('div');
    icon.className = 'flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center text-white font-bold shadow-lg';
    icon.textContent = role === 'user' ? 'U' : 'A';
    icon.classList.add(role === 'user' ? 'bg-gray-700' : 'bg-[#F15F22]');

    const messageContainer = document.createElement('div');
    messageContainer.className = 'p-4 rounded-xl shadow-lg max-w-2xl glass-panel';
    messageContainer.classList.add(role === 'user' ? 'bg-gray-800/50' : 'bg-[#333333]/50');

    const author = document.createElement('p');
    author.className = 'font-bold mb-2 text-sm';
    author.textContent = role === 'user' ? 'You' : 'Assistant';
    author.classList.add(role === 'user' ? 'text-gray-300' : 'text-[#F15F22]');
    messageContainer.appendChild(author);

    const messageContent = document.createElement('div');
    messageContent.innerHTML = content;
    messageContainer.appendChild(messageContent);

    wrapper.appendChild(role === 'user' ? messageContainer : icon);
    wrapper.appendChild(role === 'user' ? icon : messageContainer);

    chatLog.appendChild(wrapper);

    const chartContainers = messageContent.querySelectorAll('.chart-render-target');
    chartContainers.forEach(container => {
        if (container.dataset.spec) {
            renderChart(container.id, container.dataset.spec);
        }
    });

    wrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function toggleLoading(isLoading) {
    userInput.disabled = isLoading;
    submitButton.disabled = isLoading;
    newChatButton.disabled = isLoading;
    sendIcon.classList.toggle('hidden', isLoading);
    loadingSpinner.classList.toggle('hidden', !isLoading);
}

function updateStatusWindow(eventData, isFinal = false) {
    const { step, details, type } = eventData;
    if (!step) {
        return;
    }

    const lastStep = document.getElementById(`status-step-${currentStatusId}`);
    if (lastStep) {
        lastStep.classList.remove('active');
        lastStep.classList.add('completed');
        if (isInFastPath) {
            lastStep.classList.add('plan-optimization');
        }
    }

    if (type === 'plan_optimization') {
        isInFastPath = true;
    } else if (step.includes("Looping Phase") && step.includes("Complete")) {
        isInFastPath = false;
    }

    currentStatusId++;
    const stepEl = document.createElement('div');
    stepEl.id = `status-step-${currentStatusId}`;
    stepEl.className = 'status-step p-3 rounded-md';

    const stepTitle = document.createElement('h4');
    stepTitle.className = 'font-bold text-sm text-white mb-2';
    stepTitle.textContent = step;
    stepEl.appendChild(stepTitle);

    const metricsEl = document.createElement('div');
    metricsEl.className = 'per-call-metrics text-xs text-gray-400 mb-2 hidden';
    stepEl.appendChild(metricsEl);

    if (details) {
        if (typeof details === 'object' && details !== null && 'summary' in details && 'full_text' in details) {
            if (details.full_text.length > 150) {
                const detailsEl = document.createElement('details');
                detailsEl.className = 'text-xs';

                const summaryEl = document.createElement('summary');
                summaryEl.className = 'cursor-pointer text-gray-400 hover:text-white';
                summaryEl.textContent = details.summary;
                detailsEl.appendChild(summaryEl);

                const pre = document.createElement('pre');
                pre.className = 'mt-2 p-2 bg-gray-900/70 rounded-md text-gray-300 overflow-x-auto whitespace-pre-wrap';
                pre.textContent = details.full_text;
                detailsEl.appendChild(pre);

                stepEl.appendChild(detailsEl);
            } else {
                const p = document.createElement('p');
                p.className = 'text-xs text-gray-400';
                p.textContent = details.full_text;
                stepEl.appendChild(p);
            }
        } else {
            const pre = document.createElement('pre');
            pre.className = 'p-2 bg-gray-900/70 rounded-md text-xs text-gray-300 overflow-x-auto whitespace-pre-wrap';
            try {
                const parsed = typeof details === 'string' ? JSON.parse(details) : details;
                pre.textContent = JSON.stringify(parsed, null, 2);
            } catch (e) {
                pre.textContent = details;
            }
            stepEl.appendChild(pre);
        }
    }

    statusWindowContent.appendChild(stepEl);

    if (type === 'workaround') {
        stepEl.classList.add('workaround');
    } else if (type === 'error') {
        stepEl.classList.add('error');
    } else if (isInFastPath) {
        stepEl.classList.add('plan-optimization');
    }

    if (!isFinal) {
        stepEl.classList.add('active');
    } else {
        stepEl.classList.add('completed');
    }

    if (!isMouseOverStatus) {
        statusWindowContent.scrollTop = statusWindowContent.scrollHeight;
    }
}

function updateTokenDisplay(data) {
    const normalDisplay = document.getElementById('token-normal-display');
    const awsMessage = document.getElementById('token-aws-message');

    if (currentProvider === 'Amazon') {
        normalDisplay.classList.add('hidden');
        awsMessage.classList.remove('hidden');
        return;
    }

    normalDisplay.classList.remove('hidden');
    awsMessage.classList.add('hidden');

    document.getElementById('statement-input-tokens').textContent = (data.statement_input || 0).toLocaleString();
    document.getElementById('statement-output-tokens').textContent = (data.statement_output || 0).toLocaleString();
    document.getElementById('total-input-tokens').textContent = (data.total_input || 0).toLocaleString();
    document.getElementById('total-output-tokens').textContent = (data.total_output || 0).toLocaleString();
}

// --- MODIFICATION START: Simplify function to only control text indicator ---
function setThinkingIndicator(isThinking) {
    if (isThinking) {
        promptNameDisplay.classList.add('hidden');
        thinkingIndicator.classList.remove('hidden');
        thinkingIndicator.classList.add('flex');
    } else {
        thinkingIndicator.classList.add('hidden');
        thinkingIndicator.classList.remove('flex');
        promptNameDisplay.classList.remove('hidden');
    }
}
// --- MODIFICATION END ---

function updateStatusPromptName() {
    const promptNameDiv = document.getElementById('prompt-name-display');
    if (currentProvider && currentModel) {
        const isCustom = isPromptCustomForModel(currentProvider, currentModel);
        const promptType = isCustom ? `Custom` : `Default`;
        promptNameDiv.innerHTML = `
            <span class="font-semibold text-gray-300">${promptType} Prompt</span>
            <span class="text-gray-500">/</span>
            <span class="font-mono text-teradata-orange text-xs">${getNormalizedModelId(currentModel)}</span>
        `;
    } else {
        promptNameDiv.innerHTML = '<span>No Model/Prompt Loaded</span>';
    }
}

async function startStream(endpoint, body) {
    if (body.message) {
        addMessage('user', body.message);
    } else {
        addMessage('user', `Executing prompt: ${body.prompt_name}`);
    }
    userInput.value = '';
    toggleLoading(true);
    statusWindowContent.innerHTML = '';
    currentStatusId = 0;
    isInFastPath = false;
    setThinkingIndicator(false);

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const messages = buffer.split('\n\n');
            buffer = messages.pop();

            for (const message of messages) {
                if (!message) continue;

                let eventName = 'message';
                let dataLine = '';

                const lines = message.split('\n');
                for(const line of lines) {
                    if (line.startsWith('data:')) {
                        dataLine = line.substring(5).trim();
                    } else if (line.startsWith('event:')) {
                        eventName = line.substring(6).trim();
                    }
                }

                if (dataLine) {
                    const eventData = JSON.parse(dataLine);

                    // --- MODIFICATION START: Comprehensive status indicator handler ---
                    if (eventName === 'status_indicator_update') {
                        const { target, state } = eventData;
                        
                        let dot;
                        let timeout;

                        if (target === 'db') {
                            dot = teradataStatusDot;
                            timeout = mcpIndicatorTimeout;
                        } else if (target === 'llm') {
                            dot = llmStatusDot;
                            timeout = llmIndicatorTimeout;
                        } else if (target === 'context') {
                            dot = contextStatusDot;
                            timeout = contextIndicatorTimeout;
                        }

                        if (target === 'llm') {
                            setThinkingIndicator(state === 'busy');
                        }

                        if (dot) {
                            if (timeout) clearTimeout(timeout);
                            
                            if (state === 'busy') {
                                const activeClass = target === 'context' ? 'context-active' : 'busy';
                                dot.classList.remove('idle', 'connected', 'busy', 'context-active');
                                dot.classList.add(activeClass);
                            } else { // idle state
                                const idleClass = target === 'db' ? 'connected' : 'idle';
                                timeout = setTimeout(() => {
                                    dot.classList.remove('busy', 'context-active');
                                    dot.classList.add(idleClass);
                                }, 150);

                                if (target === 'db') mcpIndicatorTimeout = timeout;
                                else if (target === 'llm') llmIndicatorTimeout = timeout;
                                else if (target === 'context') contextIndicatorTimeout = timeout;
                            }
                        }
                    // --- MODIFICATION END ---
                    } else if (eventName === 'token_update') {
                        updateTokenDisplay(eventData);
                        const lastStep = document.getElementById(`status-step-${currentStatusId}`);
                        if (lastStep && currentProvider !== 'Amazon') {
                            const metricsEl = lastStep.querySelector('.per-call-metrics');
                            if (metricsEl) {
                                metricsEl.innerHTML = `(LLM Call: ${eventData.statement_input.toLocaleString()} in / ${eventData.statement_output.toLocaleString()} out)`;
                                metricsEl.classList.remove('hidden');
                            }
                         }
                    } else if (eventName === 'request_user_input') {
                        updateStatusWindow({ step: "Action Required", details: "Waiting for user to correct parameters.", type: 'workaround' });
                        toggleLoading(false);
                        openCorrectionModal(eventData.details);
                    } else if (eventName === 'session_update') {
                        const { id, name } = eventData.session_name_update;
                        const sessionItem = document.getElementById(`session-${id}`);
                        if (sessionItem) {
                            sessionItem.querySelector('span').textContent = name;
                        }
                    } else if (eventName === 'llm_thought') {
                        updateStatusWindow({ step: "Assistant's Thought Process", ...eventData });
                    } else if (eventName === 'prompt_selected') {
                        updateStatusWindow(eventData);
                        if (eventData.prompt_name) {
                            highlightResource(eventData.prompt_name, 'prompts');
                        }
                    } else if (eventName === 'tool_result') {
                        updateStatusWindow(eventData);
                        if (eventData.tool_name) {
                            const toolType = eventData.tool_name.startsWith('generate_') ? 'charts' : 'tools';
                            highlightResource(eventData.tool_name, toolType);
                        }
                    } else if (eventName === 'final_answer') {
                        addMessage('assistant', eventData.final_answer);
                        updateStatusWindow({ step: "Finished", details: "Response sent to chat." }, true);
                        toggleLoading(false);
                    } else if (eventName === 'error') {
                        addMessage('assistant', `Sorry, an error occurred: ${eventData.error}`);
                        updateStatusWindow({ step: "Error", ...eventData, type: 'error' }, true);
                        toggleLoading(false);
                    } else {
                        updateStatusWindow(eventData);
                    }
                }
            }
        }

    } catch (error) {
        addMessage('assistant', `Sorry, a connection error occurred: ${error.message}`);
        updateStatusWindow({ step: "Error", details: error.stack, type: 'error' }, true);
    } finally {
        toggleLoading(false);
        setThinkingIndicator(false);
    }
}

async function togglePrompt(promptName, isDisabled, buttonEl) {
    try {
        const res = await fetch('/prompt/toggle_status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: promptName, disabled: isDisabled })
        });

        if (!res.ok) {
            throw new Error('Server responded with an error.');
        }

        for (const category in resourceData.prompts) {
            const prompt = resourceData.prompts[category].find(p => p.name === promptName);
            if (prompt) {
                prompt.disabled = isDisabled;
                break;
            }
        }

        const promptItem = document.getElementById(`resource-prompts-${promptName}`);
        const runButton = promptItem.querySelector('.run-prompt-button');

        promptItem.classList.toggle('opacity-60', isDisabled);
        promptItem.title = isDisabled ? 'This prompt is disabled and will not be used by the agent.' : '';
        runButton.disabled = isDisabled;
        runButton.title = isDisabled ? 'This prompt is disabled.' : 'Run this prompt.';

        buttonEl.innerHTML = isDisabled ?
            `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M3.707 2.293a1 1 0 00-1.414 1.414l14 14a1 1 0 001.414-1.414l-1.473-1.473A10.014 10.014 0 0019.542 10C18.268 5.943 14.478 3 10 3a9.958 9.958 0 00-4.512 1.074L3.707 2.293zM10 12a2 2 0 110-4 2 2 0 010 4z" clip-rule="evenodd" /><path d="M2 10s3.939 4 8 4 8-4 8-4-3.939-4-8-4-8 4-8 4zm13.707 4.293a1 1 0 00-1.414-1.414L12.586 14.6A8.007 8.007 0 0110 16c-4.478 0-8.268-2.943-9.542-7 .946-2.317 2.83-4.224 5.166-5.447L2.293 1.293A1 1 0 00.879 2.707l14 14a1 1 0 001.414 0z" /></svg>` :
            `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M10 12a2 2 0 100-4 2 2 0 000 4z" /><path fill-rule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.022 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clip-rule="evenodd" /></svg>`;

        updatePromptsTabCounter();

    } catch (error) {
        console.error(`Failed to toggle prompt ${promptName}:`, error);
    }
}

async function toggleTool(toolName, isDisabled, buttonEl) {
    try {
        const res = await fetch('/tool/toggle_status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: toolName, disabled: isDisabled })
        });

        if (!res.ok) {
            throw new Error('Server responded with an error.');
        }

        for (const category in resourceData.tools) {
            const tool = resourceData.tools[category].find(t => t.name === toolName);
            if (tool) {
                tool.disabled = isDisabled;
                break;
            }
        }

        const toolItem = document.getElementById(`resource-tools-${toolName}`);
        toolItem.classList.toggle('opacity-60', isDisabled);
        toolItem.title = isDisabled ? 'This tool is disabled and will not be used by the agent.' : '';

        buttonEl.innerHTML = isDisabled ?
            `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M3.707 2.293a1 1 0 00-1.414 1.414l14 14a1 1 0 001.414-1.414l-1.473-1.473A10.014 10.014 0 0019.542 10C18.268 5.943 14.478 3 10 3a9.958 9.958 0 00-4.512 1.074L3.707 2.293zM10 12a2 2 0 110-4 2 2 0 010 4z" clip-rule="evenodd" /><path d="M2 10s3.939 4 8 4 8-4 8-4-3.939-4-8-4-8 4-8 4zm13.707 4.293a1 1 0 00-1.414-1.414L12.586 14.6A8.007 8.007 0 0110 16c-4.478 0-8.268-2.943-9.542-7 .946-2.317 2.83-4.224 5.166-5.447L2.293 1.293A1 1 0 00.879 2.707l14 14a1 1 0 001.414 0z" /></svg>` :
            `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M10 12a2 2 0 100-4 2 2 0 000 4z" /><path fill-rule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.022 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clip-rule="evenodd" /></svg>`;

        updateToolsTabCounter();

    } catch (error) {
        console.error(`Failed to toggle tool ${toolName}:`, error);
    }
}

function createResourceItem(resource, type) {
    const detailsEl = document.createElement('details');
    detailsEl.id = `resource-${type}-${resource.name}`;
    detailsEl.className = 'resource-item bg-gray-800/50 rounded-lg border border-gray-700/60';

    if (resource.disabled) {
        detailsEl.classList.add('opacity-60');
        detailsEl.title = `This ${type.slice(0, -1)} is disabled and will not be used by the agent.`;
    }

    const toggleIcon = resource.disabled ?
        `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M3.707 2.293a1 1 0 00-1.414 1.414l14 14a1 1 0 001.414-1.414l-1.473-1.473A10.014 10.014 0 0019.542 10C18.268 5.943 14.478 3 10 3a9.958 9.958 0 00-4.512 1.074L3.707 2.293zM10 12a2 2 0 110-4 2 2 0 010 4z" clip-rule="evenodd" /><path d="M2 10s3.939 4 8 4 8-4 8-4-3.939-4-8-4-8 4-8 4zm13.707 4.293a1 1 0 00-1.414-1.414L12.586 14.6A8.007 8.007 0 0110 16c-4.478 0-8.268-2.943-9.542-7 .946-2.317 2.83-4.224 5.166-5.447L2.293 1.293A1 1 0 00.879 2.707l14 14a1 1 0 001.414 0z" /></svg>` :
        `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M10 12a2 2 0 100-4 2 2 0 000 4z" /><path fill-rule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.022 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clip-rule="evenodd" /></svg>`;

    let contentHTML = '';
    if (type === 'prompts') {
        const runButtonDisabledAttr = resource.disabled ? 'disabled' : '';
        const runButtonTitle = resource.disabled ? 'This prompt is disabled.' : 'Run this prompt.';

        let argsHTML = '';
        if (resource.arguments && resource.arguments.length > 0) {
            argsHTML += `<div class="mt-4 pt-3 border-t border-gray-700/60">
                                    <h5 class="font-semibold text-sm text-white mb-2">Parameters</h5>
                                    <ul class="space-y-2 text-xs">`;
            resource.arguments.forEach(arg => {
                const requiredText = arg.required ? '<span class="text-red-400 font-bold">Required</span>' : '<span class="text-gray-400">Optional</span>';
                const typeText = arg.type && arg.type !== 'unknown' ? `<span class="font-mono text-xs text-cyan-400 bg-cyan-400/10 px-1.5 py-0.5 rounded-md">${arg.type}</span>` : '';

                argsHTML += `<li class="p-2 bg-black/20 rounded-md">
                                        <div class="flex justify-between items-center">
                                            <div class="flex items-center gap-x-2">
                                                <code class="font-semibold text-teradata-orange">${arg.name}</code>
                                                ${typeText}
                                            </div>
                                            ${requiredText}
                                        </div>
                                        <p class="text-gray-400 mt-1">${arg.description}</p>
                                     </li>`;
            });
            argsHTML += `</ul></div>`;
        }

        contentHTML = `
                <div class="p-3 pt-2 text-sm text-gray-300 space-y-3">
                    <p>${resource.description}</p>
                    ${argsHTML}
                    <div class="flex justify-end items-center gap-x-2 pt-3 border-t border-gray-700/60">
                        <button class="prompt-toggle-button p-1.5 text-gray-300 hover:text-white hover:bg-white/10 rounded-md transition-colors">${toggleIcon}</button>
                        <button class="view-prompt-button px-3 py-1 bg-gray-600 text-white text-xs font-semibold rounded-md hover:bg-gray-500 transition-colors">Prompt</button>
                        <button class="run-prompt-button px-3 py-1 bg-teradata-orange text-white text-xs font-semibold rounded-md hover:bg-teradata-orange-dark transition-colors" ${runButtonDisabledAttr} title="${runButtonTitle}">Run</button>
                    </div>
                </div>`;
    } else { // For tools and resources
         contentHTML = `
                <div class="p-3 pt-2 text-sm text-gray-300 border-t border-gray-700/60 flex justify-between items-center">
                    <p>${resource.description}</p>
                    ${type === 'tools' ? `<button class="tool-toggle-button p-1.5 text-gray-300 hover:text-white hover:bg-white/10 rounded-md transition-colors">${toggleIcon}</button>` : ''}
                </div>`;
    }

    detailsEl.innerHTML = `
                <summary class="flex justify-between items-center p-3 font-semibold text-white hover:bg-gray-700/50 rounded-lg transition-colors cursor-pointer">
                    <span>${resource.name}</span>
                    <svg class="chevron w-5 h-5 text-[#F15F22] flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
                </summary>
                ${contentHTML}
            `;

    if (type === 'prompts') {
        const runButton = detailsEl.querySelector('.run-prompt-button');
        runButton.addEventListener('click', (e) => { e.stopPropagation(); if (!runButton.disabled) openPromptModal(resource); });
        const viewButton = detailsEl.querySelector('.view-prompt-button');
        viewButton.addEventListener('click', (e) => { e.stopPropagation(); openViewPromptModal(resource.name); });
        const toggleButton = detailsEl.querySelector('.prompt-toggle-button');
        toggleButton.addEventListener('click', (e) => { e.stopPropagation(); togglePrompt(resource.name, !resource.disabled, toggleButton); });
    } else if (type === 'tools') {
        const toggleButton = detailsEl.querySelector('.tool-toggle-button');
        if(toggleButton) {
            toggleButton.addEventListener('click', (e) => { e.stopPropagation(); toggleTool(resource.name, !resource.disabled, toggleButton); });
        }
    }

    return detailsEl;
}

function updatePromptsTabCounter() {
    const tabButton = document.querySelector('.resource-tab[data-type="prompts"]');
    if (!tabButton || !resourceData.prompts) return;
    let totalCount = 0;
    let disabledCount = 0;
    Object.values(resourceData.prompts).forEach(items => {
        totalCount += items.length;
        disabledCount += items.filter(item => item.disabled).length;
    });
    const disabledIndicator = disabledCount > 0 ? '*' : '';
    tabButton.textContent = `Prompts (${totalCount})${disabledIndicator}`;
}

function updateToolsTabCounter() {
    const tabButton = document.querySelector('.resource-tab[data-type="tools"]');
    if (!tabButton || !resourceData.tools) return;
    let totalCount = 0;
    let disabledCount = 0;
    Object.values(resourceData.tools).forEach(items => {
        totalCount += items.length;
        disabledCount += items.filter(item => item.disabled).length;
    });
    const disabledIndicator = disabledCount > 0 ? '*' : '';
    tabButton.textContent = `Tools (${totalCount})${disabledIndicator}`;
}

async function loadResources(type) {
    const tabButton = document.querySelector(`.resource-tab[data-type="${type}"]`);
    const categoriesContainer = document.getElementById(`${type}-categories`);
    const panelsContainer = document.getElementById(`${type}-panels-container`);
    const typeCapitalized = type.charAt(0).toUpperCase() + type.slice(1);

    try {
        const res = await fetch(`/${type}`);
        const data = await res.json();

        if (!res.ok || data.error || Object.keys(data).length === 0) {
            if(tabButton) tabButton.style.display = 'none';
            return;
        }

        tabButton.style.display = 'inline-block';
        resourceData[type] = data;

        if (type === 'prompts') {
            updatePromptsTabCounter();
        } else if (type === 'tools') {
            updateToolsTabCounter();
        } else {
            const totalCount = Object.values(data).reduce((acc, items) => acc + items.length, 0);
            tabButton.textContent = `${typeCapitalized} (${totalCount})`;
        }

        categoriesContainer.innerHTML = '';
        panelsContainer.innerHTML = '';

        Object.keys(data).forEach(category => {
            const categoryTab = document.createElement('button');
            categoryTab.className = 'category-tab px-4 py-2 rounded-md font-semibold text-sm transition-colors hover:bg-[#D9501A]';
            categoryTab.textContent = category;
            categoryTab.dataset.category = category;
            categoryTab.dataset.type = type;
            categoriesContainer.appendChild(categoryTab);

            const panel = document.createElement('div');
            panel.id = `panel-${type}-${category}`;
            panel.className = 'category-panel px-4 space-y-2';
            panel.dataset.category = category;

            data[category].forEach(resource => {
                const itemEl = createResourceItem(resource, type);
                panel.appendChild(itemEl);
            });
            panelsContainer.appendChild(panel);
        });

        document.querySelectorAll(`#${type}-categories .category-tab`).forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll(`#${type}-categories .category-tab`).forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                document.querySelectorAll(`#${type}-panels-container .category-panel`).forEach(p => {
                    p.classList.toggle('open', p.dataset.category === tab.dataset.category);
                });
            });
        });

        if (categoriesContainer.querySelector('.category-tab')) {
            categoriesContainer.querySelector('.category-tab').click();
        }

    } catch (error) {
        console.error(`Failed to load ${type}: ${error.message}`);
        if(tabButton) {
            tabButton.textContent = `${typeCapitalized} (Error)`;
            tabButton.style.display = 'inline-block';
        }
        categoriesContainer.innerHTML = '';
        panelsContainer.innerHTML = `<div class="p-4 text-center text-red-400">Failed to load ${type}.</div>`;
    }
}

function highlightResource(resourceName, type) {
    if (currentlySelectedResource) {
        currentlySelectedResource.classList.remove('resource-selected');
    }

    let resourceCategory = null;
    for (const category in resourceData[type]) {
        if (resourceData[type][category].some(r => r.name === resourceName)) {
            resourceCategory = category;
            break;
        }
    }

    if (resourceCategory) {
        document.querySelector(`.resource-tab[data-type="${type}"]`).click();
        const categoryTab = document.querySelector(`.category-tab[data-type="${type}"][data-category="${resourceCategory}"]`);
        if(categoryTab) categoryTab.click();

        const resourceElement = document.getElementById(`resource-${type}-${resourceName}`);
        if (resourceElement) {
            resourceElement.open = true;
            resourceElement.classList.add('resource-selected');
            currentlySelectedResource = resourceElement;

            setTimeout(() => {
                resourceElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 350);
        }
    }
}

async function startNewSession() {
    const activePrompt = getSystemPromptForModel(currentProvider, currentModel);
    if (!activePrompt) {
        addMessage('assistant', 'Cannot start a new session. The system prompt is not loaded for the current model. Please re-configure.');
        return;
    }
    chatLog.innerHTML = '';
    statusWindowContent.innerHTML = '<p class="text-gray-400">Waiting for a new request...</p>';
    updateTokenDisplay({ statement_input: 0, statement_output: 0, total_input: 0, total_output: 0 });
    addMessage('assistant', "Starting a new conversation... Please wait.");
    toggleLoading(true);
    setThinkingIndicator(false);
    try {
        const res = await fetch('/session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                system_prompt: activePrompt,
                charting_intensity: chartingIntensitySelect.value
            })
        });
        const data = await res.json();
        if (res.ok && data.session_id) {
            addSessionToList(data.session_id, data.name, true);
            loadSession(data.session_id);
        } else {
            throw new Error(data.error || "Failed to get a session ID.");
        }
    } catch (error) {
        addMessage('assistant', `Failed to start a new session: ${error.message}`);
    } finally {
        toggleLoading(false);
        userInput.focus();
    }
}

async function loadSession(sessionId) {
    if (currentSessionId === sessionId) return;

    toggleLoading(true);
    try {
        const res = await fetch(`/session/${sessionId}`);
        const data = await res.json();

        if (res.ok) {
            currentSessionId = sessionId;
            chatLog.innerHTML = '';
            data.history.forEach(msg => addMessage(msg.role, msg.content));
            updateTokenDisplay({ total_input: data.input_tokens, total_output: data.output_tokens });

            document.querySelectorAll('.session-item').forEach(item => {
                item.classList.toggle('active', item.dataset.sessionId === sessionId);
            });

            if (data.history.length === 0) {
                 addMessage('assistant', "I'm ready to help. How can I assist you with your Teradata system today?");
            }
            updateStatusPromptName();
        } else {
            addMessage('assistant', `Error loading session: ${data.error}`);
        }
    } catch (error) {
        addMessage('assistant', `Failed to load session: ${error.message}`);
    } finally {
        toggleLoading(false);
        userInput.focus();
    }
}

function addSessionToList(sessionId, name, isActive = false) {
    const sessionItem = document.createElement('button');
    sessionItem.id = `session-${sessionId}`;
    sessionItem.dataset.sessionId = sessionId;
    sessionItem.className = 'session-item w-full text-left p-3 rounded-lg hover:bg-white/10 transition-colors truncate';
    if (isActive) {
        document.querySelectorAll('.session-item').forEach(item => item.classList.remove('active'));
        sessionItem.classList.add('active');
    }

    const nameSpan = document.createElement('span');
    nameSpan.className = 'font-semibold text-sm text-white';
    nameSpan.textContent = name;
    sessionItem.appendChild(nameSpan);

    sessionItem.addEventListener('click', () => loadSession(sessionId));
    sessionList.prepend(sessionItem);
}

async function loadAllSessions() {
    try {
        const res = await fetch('/sessions');
        const sessions = await res.json();
        sessionList.innerHTML = '';
        if (sessions && sessions.length > 0) {
            sessions.forEach(s => addSessionToList(s.id, s.name));
            loadSession(sessions[0].id);
        } else {
        }
    } catch (e) {
        console.error("Could not load sessions", e);
        addMessage('assistant', 'Could not retrieve past sessions. Please configure the application to start.');
    }
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message || !currentSessionId) return;
    startStream('/ask_stream', { message, session_id: currentSessionId });
});

newChatButton.addEventListener('click', startNewSession);

resourceTabs.addEventListener('click', (e) => {
    if (e.target.classList.contains('resource-tab')) {
        const type = e.target.dataset.type;
        document.querySelectorAll('.resource-tab').forEach(tab => tab.classList.remove('active'));
        e.target.classList.add('active');

        document.querySelectorAll('.resource-panel').forEach(panel => {
            panel.style.display = panel.id === `${type}-panel` ? 'flex' : 'none';
        });
    }
});

function openPromptModal(prompt) {
    promptModalOverlay.classList.remove('hidden', 'opacity-0');
    promptModalContent.classList.remove('scale-95', 'opacity-0');
    promptModalTitle.textContent = prompt.name;
    promptModalForm.dataset.promptName = prompt.name;
    promptModalInputs.innerHTML = '';
    promptModalForm.querySelector('button[type="submit"]').textContent = 'Run Prompt';


    if (prompt.arguments && prompt.arguments.length > 0) {
        prompt.arguments.forEach(arg => {
            const inputGroup = document.createElement('div');
            const label = document.createElement('label');
            label.htmlFor = `prompt-arg-${arg.name}`;
            label.className = 'block text-sm font-medium text-gray-300 mb-1';
            label.textContent = arg.name + (arg.required ? ' *' : '');

            const input = document.createElement('input');
            input.type = 'text';
            input.id = `prompt-arg-${arg.name}`;
            input.name = arg.name;
            input.className = 'w-full p-2 bg-gray-700 border border-gray-600 rounded-md focus:ring-2 focus:ring-[#F15F22] focus:border-[#F15F22] outline-none';
            input.placeholder = arg.description || `Enter value for ${arg.name}`;
            if (arg.required) input.required = true;

            inputGroup.appendChild(label);
            inputGroup.appendChild(input);
            promptModalInputs.appendChild(inputGroup);
        });
    } else {
        promptModalInputs.innerHTML = '<p class="text-gray-400">This prompt requires no arguments.</p>';
    }

    promptModalForm.onsubmit = async (e) => {
        e.preventDefault();
        const promptName = e.target.dataset.promptName;
        const formData = new FormData(e.target);
        const arugments = Object.fromEntries(formData.entries());

        closePromptModal();
        await startStream('/invoke_prompt_stream', {
            session_id: currentSessionId,
            prompt_name: promptName,
            arguments: arugments
        });
    };
}

function openCorrectionModal(data) {
    promptModalOverlay.classList.remove('hidden', 'opacity-0');
    promptModalContent.classList.remove('scale-95', 'opacity-0');

    const spec = data.specification;
    promptModalTitle.textContent = `Correction for: ${spec.name}`;
    promptModalForm.dataset.toolName = spec.name;
    promptModalInputs.innerHTML = '';
    promptModalForm.querySelector('button[type="submit"]').textContent = 'Run Correction';

    const messageEl = document.createElement('p');
    messageEl.className = 'text-yellow-300 text-sm mb-4 p-3 bg-yellow-500/10 rounded-lg';
    messageEl.textContent = data.message;
    promptModalInputs.appendChild(messageEl);

    spec.arguments.forEach(arg => {
        const inputGroup = document.createElement('div');
        const label = document.createElement('label');
        label.htmlFor = `correction-arg-${arg.name}`;
        label.className = 'block text-sm font-medium text-gray-300 mb-1';
        label.textContent = arg.name + (arg.required ? ' *' : '');

        const input = document.createElement('input');
        input.type = 'text';
        input.id = `correction-arg-${arg.name}`;
        input.name = arg.name;
        input.className = 'w-full p-2 bg-gray-700 border border-gray-600 rounded-md focus:ring-2 focus:ring-[#F15F22] focus:border-[#F15F22] outline-none';
        input.placeholder = arg.description || `Enter value for ${arg.name}`;
        if (arg.required) input.required = true;

        inputGroup.appendChild(label);
        inputGroup.appendChild(input);
        promptModalInputs.appendChild(inputGroup);
    });

    promptModalForm.onsubmit = async (e) => {
        e.preventDefault();
        const toolName = e.target.dataset.toolName;
        const formData = new FormData(e.target);
        const userArgs = Object.fromEntries(formData.entries());

        const correctedPrompt = `Please run the tool '${toolName}' with the following corrected parameters: ${JSON.stringify(userArgs)}`;

        closePromptModal();

        startStream('/ask_stream', { message: correctedPrompt, session_id: currentSessionId });
    };
}

function closePromptModal() {
    promptModalOverlay.classList.add('opacity-0');
    promptModalContent.classList.add('scale-95', 'opacity-0');
    setTimeout(() => promptModalOverlay.classList.add('hidden'), 300);
}

promptModalClose.addEventListener('click', closePromptModal);
promptModalOverlay.addEventListener('click', (e) => {
    if (e.target === promptModalOverlay) closePromptModal();
});

async function openViewPromptModal(promptName) {
    viewPromptModalOverlay.classList.remove('hidden', 'opacity-0');
    viewPromptModalContent.classList.remove('scale-95', 'opacity-0');
    viewPromptModalTitle.textContent = `Viewing Prompt: ${promptName}`;
    viewPromptModalText.textContent = 'Loading...';

    try {
        const res = await fetch(`/prompt/${promptName}`);
        const data = await res.json();
        if (res.ok) {
            viewPromptModalText.textContent = data.content;
        } else {
            if (data.error === 'dynamic_prompt_error') {
                viewPromptModalText.textContent = `Info: ${data.message}`;
            } else {
                throw new Error(data.error || 'Failed to fetch prompt content.');
            }
        }
    } catch (error) {
        viewPromptModalText.textContent = `Error: ${error.message}`;
    }
}

function closeViewPromptModal() {
    viewPromptModalOverlay.classList.add('opacity-0');
    viewPromptModalContent.classList.add('scale-95', 'opacity-0');
    setTimeout(() => viewPromptModalOverlay.classList.add('hidden'), 300);
}

viewPromptModalClose.addEventListener('click', closeViewPromptModal);
viewPromptModalOverlay.addEventListener('click', (e) => {
    if (e.target === viewPromptModalOverlay) closeViewPromptModal();
});


function setupPanelToggle(button, panel, checkbox, collapseIcon, expandIcon) {
    const toggle = (isOpen) => {
        const isCollapsed = !isOpen;
        panel.classList.toggle('collapsed', isCollapsed);
        if (collapseIcon) collapseIcon.classList.toggle('hidden', isCollapsed);
        if (expandIcon) expandIcon.classList.toggle('hidden', !isCollapsed);
        if (checkbox) checkbox.checked = isOpen;
    };

    button.addEventListener('click', () => toggle(panel.classList.contains('collapsed')));
    if (checkbox) {
        checkbox.addEventListener('change', () => toggle(checkbox.checked));
    }
}

windowMenuButton.addEventListener('click', (e) => {
    e.stopPropagation();
    windowDropdownMenu.classList.toggle('open');
});
document.addEventListener('click', (e) => {
    if (!windowDropdownMenu.contains(e.target) && e.target !== windowMenuButton) {
        windowDropdownMenu.classList.remove('open');
    }
});

infoButton.addEventListener('click', () => {
    infoModalOverlay.classList.remove('hidden', 'opacity-0');
    infoModalContent.classList.remove('scale-95', 'opacity-0');
});
infoModalClose.addEventListener('click', () => {
    infoModalOverlay.classList.add('opacity-0');
    infoModalContent.classList.add('scale-95', 'opacity-0');
    setTimeout(() => infoModalOverlay.classList.add('hidden'), 300);
});
infoModalOverlay.addEventListener('click', (e) => {
    if (e.target === infoModalOverlay) {
        infoModalClose.click();
    }
});

function closeConfigModal() {
    configModalOverlay.classList.add('opacity-0');
    configModalContent.classList.add('scale-95', 'opacity-0');
    setTimeout(() => configModalOverlay.classList.add('hidden'), 300);
}

function getCurrentCoreConfig() {
    const formData = new FormData(configForm);
    return Object.fromEntries(formData.entries());
}

function updateConfigButtonState() {
    const isConnected = teradataStatusDot.classList.contains('connected');

    if (!isConnected) {
        configActionButtonText.textContent = 'Connect & Load';
        configActionButton.type = 'submit';
        configActionButton.classList.remove('bg-gray-600', 'hover:bg-gray-500');
        configActionButton.classList.add('bg-blue-600', 'hover:bg-blue-700');
        configActionButton.onclick = null;
        return;
    }

    const currentConfig = getCurrentCoreConfig();
    const hasChanged = JSON.stringify(currentConfig) !== JSON.stringify(pristineConfig);

    if (hasChanged) {
        configActionButtonText.textContent = 'Reconnect & Load';
        configActionButton.type = 'submit';
        configActionButton.classList.remove('bg-gray-600', 'hover:bg-gray-500');
        configActionButton.classList.add('bg-blue-600', 'hover:bg-blue-700');
        configActionButton.onclick = null;
    } else {
        configActionButtonText.textContent = 'Close';
        configActionButton.type = 'button';
        configActionButton.classList.remove('bg-blue-600', 'hover:bg-blue-700');
        configActionButton.classList.add('bg-gray-600', 'hover:bg-gray-500');
        configActionButton.onclick = closeConfigModal;
    }
}

configMenuButton.addEventListener('click', () => {
    configModalOverlay.classList.remove('hidden', 'opacity-0');
    configModalContent.classList.remove('scale-95', 'opacity-0');

    pristineConfig = getCurrentCoreConfig();
    updateConfigButtonState();
});

configModalClose.addEventListener('click', () => {
    const coreChanged = JSON.stringify(getCurrentCoreConfig()) !== JSON.stringify(pristineConfig);
    if (coreChanged) {
        showConfirmation('Discard Changes?', 'You have unsaved changes in your configuration. Are you sure you want to close?', closeConfigModal);
    } else {
        closeConfigModal();
    }
});

function startPopupCountdown() {
    if (systemPromptPopupTimer) {
        clearInterval(systemPromptPopupTimer);
    }

    const countdownTimerEl = document.getElementById('countdown-timer');
    const countdownContainerEl = document.getElementById('countdown-container');

    if (countdownTimerEl && countdownContainerEl) {
        countdownTimerEl.textContent = countdownValue;
        countdownContainerEl.style.visibility = 'visible';

        systemPromptPopupTimer = setInterval(() => {
            countdownValue--;
            countdownTimerEl.textContent = countdownValue;
            if (countdownValue <= 0) {
                closeSystemPromptPopup();
            }
        }, 1000);
    }
}

function stopPopupCountdown() {
    if (systemPromptPopupTimer) {
        clearInterval(systemPromptPopupTimer);
        systemPromptPopupTimer = null;
    }
    const countdownContainerEl = document.getElementById('countdown-container');
    if(countdownContainerEl) {
        countdownContainerEl.style.visibility = 'hidden';
    }
}

function buildDisabledCapabilitiesList() {
    const disabledTools = [];
    if (resourceData.tools) {
        Object.values(resourceData.tools).flat().forEach(tool => {
            if (tool.disabled) disabledTools.push(tool.name);
        });
    }

    const disabledPrompts = [];
    if (resourceData.prompts) {
        Object.values(resourceData.prompts).flat().forEach(prompt => {
            if (prompt.disabled) disabledPrompts.push(prompt.name);
        });
    }

    if (disabledTools.length === 0 && disabledPrompts.length === 0) {
        disabledCapabilitiesContainer.style.display = 'none';
        return '';
    }

    disabledCapabilitiesContainer.style.display = 'block';

    let html = `
        <div class="border-b border-white/10 pb-4 mb-4">
            <h4 class="text-md font-bold text-yellow-300 mb-2">Inactive Capabilities</h4>
            <p class="text-xs text-gray-400 mb-3">The following are currently disabled for the agent (due to a necessary refinement on the MCP server). You can enable them for testing in the Capabilities Panel.</p>
            <div class="flex gap-x-8">
    `;

    if (disabledTools.length > 0) {
        html += '<div><h5 class="font-semibold text-sm text-white mb-1">Tools</h5><ul class="list-disc list-inside text-xs text-gray-300 space-y-1">';
        disabledTools.forEach(name => {
            html += `<li><code class="text-teradata-orange text-xs">${name}</code></li>`;
        });
        html += '</ul></div>';
    }

    if (disabledPrompts.length > 0) {
        html += '<div><h5 class="font-semibold text-sm text-white mb-1">Prompts</h5><ul class="list-disc list-inside text-xs text-gray-300 space-y-1">';
        disabledPrompts.forEach(name => {
            html += `<li><code class="text-teradata-orange text-xs">${name}</code></li>`;
        });
        html += '</ul></div>';
    }

    html += '</div></div>';
    return html;
}

function openSystemPromptPopup() {
    const promptText = getSystemPromptForModel(currentProvider, currentModel);
    const isCustom = isPromptCustomForModel(currentProvider, currentModel);

    if (isCustom) {
        systemPromptPopupTitle.innerHTML = `Custom System Prompt for: <code class="text-teradata-orange font-normal">${currentProvider} / ${getNormalizedModelId(currentModel)}</code>`;
    } else {
        systemPromptPopupTitle.innerHTML = `Default System Prompt Selected for: <code class="text-teradata-orange font-normal">${currentProvider}</code>`;
    }

    disabledCapabilitiesContainer.innerHTML = buildDisabledCapabilitiesList();
    systemPromptPopupText.textContent = promptText || "Could not load system prompt.";

    systemPromptPopupOverlay.classList.remove('hidden', 'opacity-0');
    systemPromptPopupContent.classList.remove('scale-95', 'opacity-0');

    countdownValue = 5;
    startPopupCountdown();

    mouseMoveHandler = () => {
        stopPopupCountdown();
        document.removeEventListener('mousemove', mouseMoveHandler);
    };
    document.addEventListener('mousemove', mouseMoveHandler);
}

function closeSystemPromptPopup() {
    stopPopupCountdown();
    if (mouseMoveHandler) {
         document.removeEventListener('mousemove', mouseMoveHandler);
         mouseMoveHandler = null;
    }
    systemPromptPopupOverlay.classList.add('opacity-0');
    systemPromptPopupContent.classList.add('scale-95', 'opacity-0');
    setTimeout(() => {
        systemPromptPopupOverlay.classList.add('hidden');
    }, 300);
}

systemPromptPopupClose.addEventListener('click', closeSystemPromptPopup);
systemPromptPopupOverlay.addEventListener('click', (e) => {
    if (e.target === systemPromptPopupOverlay) closeSystemPromptPopup();
});

configForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const selectedModel = llmModelSelect.value;
    if (!selectedModel) {
        configStatus.textContent = 'Please select your LLM Model.';
        configStatus.className = 'text-sm text-red-400 text-center';
        return;
    }

    configLoadingSpinner.classList.remove('hidden');
    configActionButton.disabled = true;
    configStatus.textContent = 'Connecting to Teradata & LLM...';
    configStatus.className = 'text-sm text-yellow-400 text-center';

    const formData = new FormData(e.target);
    const config = Object.fromEntries(formData.entries());

    const mcpConfig = { host: config.host, port: config.port, path: config.path };
    localStorage.setItem('mcpConfig', JSON.stringify(mcpConfig));

    if (config.provider === 'Amazon') {
        const awsCreds = { aws_access_key_id: config.aws_access_key_id, aws_secret_access_key: config.aws_secret_access_key, aws_region: config.aws_region };
        localStorage.setItem('amazonApiKey', JSON.stringify(awsCreds));
    } else if (config.provider === 'Ollama') {
        localStorage.setItem('ollamaHost', config.ollama_host);
    } else {
        localStorage.setItem(`${config.provider.toLowerCase()}ApiKey`, config.apiKey);
    }

    try {
        const res = await fetch('/configure', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const result = await res.json();

        if (res.ok) {
            configStatus.textContent = 'Success! Teradata & LLM services connected.';
            configStatus.className = 'text-sm text-green-400 text-center';
            teradataStatusDot.classList.remove('disconnected');
            teradataStatusDot.classList.add('connected');
            llmStatusDot.classList.remove('disconnected', 'busy');
            llmStatusDot.classList.add('connected');
            contextStatusDot.classList.remove('disconnected');
            contextStatusDot.classList.add('idle');

            localStorage.setItem('lastSelectedProvider', config.provider);

            currentProvider = config.provider;
            currentModel = config.model;

            const activePrompt = getSystemPromptForModel(currentProvider, currentModel);
            if (!activePrompt) {
                await resetSystemPrompt(true);
            }

            chatModalButton.disabled = false;
            promptEditorButton.disabled = false;

            await loadResources('tools');
            await loadResources('prompts');
            await loadResources('resources');

            userInput.placeholder = "Ask about databases, tables, users...";

            await startNewSession();

            pristineConfig = getCurrentCoreConfig();

            setTimeout(closeConfigModal, 1000);

            updateConfigButtonState();
            openSystemPromptPopup();

        } else {
            throw new Error(result.message || 'An unknown configuration error occurred.');
        }
    } catch (error) {
        configStatus.textContent = `Error: ${error.message}`;
        configStatus.className = 'text-sm text-red-400 text-center';
        promptEditorButton.disabled = true;
        chatModalButton.disabled = true;
        teradataStatusDot.classList.add('disconnected');
        teradataStatusDot.classList.remove('connected');
        llmStatusDot.classList.add('disconnected');
        llmStatusDot.classList.remove('connected', 'idle');
        contextStatusDot.classList.add('disconnected');
        contextStatusDot.classList.remove('idle', 'context-active');
    } finally {
        configLoadingSpinner.classList.add('hidden');
        configActionButton.disabled = false;
        updateConfigButtonState();
    }
});

async function fetchModels() {
    const provider = llmProviderSelect.value;
    let body = { provider };

    if (provider === 'Amazon') {
        body.aws_access_key_id = awsAccessKeyIdInput.value;
        body.aws_secret_access_key = awsSecretAccessKeyInput.value;
        body.aws_region = awsRegionInput.value;
        body.listing_method = document.querySelector('input[name="listing_method"]:checked').value;
    } else if (provider === 'Ollama') {
        body.host = ollamaHostInput.value;
    } else {
        body.apiKey = llmApiKeyInput.value;
    }

    if (
        (provider === 'Amazon' && (!body.aws_access_key_id || !body.aws_secret_access_key || !body.aws_region)) ||
        (provider === 'Ollama' && !body.host) ||
        (!['Amazon', 'Ollama'].includes(provider) && !body.apiKey)
    ) {
         configStatus.textContent = 'API credentials or host are required to fetch models.';
        configStatus.className = 'text-sm text-yellow-400 text-center';
        return;
    }

    refreshIcon.classList.add('hidden');
    refreshSpinner.classList.remove('hidden');
    refreshModelsButton.disabled = true;
    configStatus.textContent = 'Fetching models...';
    configStatus.className = 'text-sm text-gray-400 text-center';

    try {
        const response = await fetch('/models', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const result = await response.json();

        if (response.ok) {
            llmModelSelect.innerHTML = '';
            result.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = model.name + (model.certified ? '' : ' (support evaluated)');
                option.disabled = !model.certified;
                llmModelSelect.appendChild(option);
            });
            configStatus.textContent = `Successfully fetched ${result.models.length} models.`;
            configStatus.className = 'text-sm text-green-400 text-center';

            if (llmModelSelect.value) {
                await handleModelChange();
            }

        } else {
            throw new Error(result.message || 'Failed to fetch models.');
        }
    } catch (error) {
        configStatus.textContent = `Error: ${error.message}`;
        configStatus.className = 'text-sm text-red-400 text-center';
        llmModelSelect.innerHTML = '<option value="">-- Could not fetch models --</option>';
    } finally {
        refreshIcon.classList.remove('hidden');
        refreshSpinner.classList.add('hidden');
        refreshModelsButton.disabled = false;
    }
}

async function handleModelChange() {
    currentModel = llmModelSelect.value;
    currentProvider = llmProviderSelect.value;
    if (!currentModel || !currentProvider) return;

    const activePrompt = getSystemPromptForModel(currentProvider, currentModel);
    if (!activePrompt) {
        configStatus.textContent = `Fetching default prompt for ${getNormalizedModelId(currentModel)}...`;
        configStatus.className = 'text-sm text-gray-400 text-center';
        await resetSystemPrompt(true);
        configStatus.textContent = `Default prompt for ${getNormalizedModelId(currentModel)} loaded.`;
        setTimeout(() => { configStatus.textContent = ''; }, 2000);
    }
}

llmProviderSelect.addEventListener('change', async () => {
    const newProvider = llmProviderSelect.value;

    apiKeyContainer.classList.add('hidden');
    awsCredentialsContainer.classList.add('hidden');
    awsListingMethodContainer.classList.add('hidden');
    ollamaHostContainer.classList.add('hidden');

    if (newProvider === 'Amazon') {
        awsCredentialsContainer.classList.remove('hidden');
        awsListingMethodContainer.classList.remove('hidden');

        const res = await fetch('/api_key/amazon');
        const envCreds = await res.json();
        const savedCreds = JSON.parse(localStorage.getItem('amazonApiKey')) || {};

        awsAccessKeyIdInput.value = envCreds.aws_access_key_id || savedCreds.aws_access_key_id || '';
        awsSecretAccessKeyInput.value = envCreds.aws_secret_access_key || savedCreds.aws_secret_access_key || '';
        awsRegionInput.value = envCreds.aws_region || savedCreds.aws_region || '';

    } else if (newProvider === 'Ollama') {
        ollamaHostContainer.classList.remove('hidden');
        const res = await fetch('/api_key/ollama');
        const data = await res.json();
        ollamaHostInput.value = data.host || localStorage.getItem('ollamaHost') || 'http://localhost:11434';
    } else {
        apiKeyContainer.classList.remove('hidden');
        const res = await fetch(`/api_key/${newProvider.toLowerCase()}`);
        const data = await res.json();
        llmApiKeyInput.value = data.apiKey || localStorage.getItem(`${newProvider.toLowerCase()}ApiKey`) || '';
    }
    llmModelSelect.innerHTML = '<option value="">-- Select Provider & Enter Credentials --</option>';
    configStatus.textContent = '';
});

[awsAccessKeyIdInput, awsSecretAccessKeyInput, awsRegionInput].forEach(input => {
    input.addEventListener('blur', () => {
        const awsCreds = {
            aws_access_key_id: awsAccessKeyIdInput.value,
            aws_secret_access_key: awsSecretAccessKeyInput.value,
            aws_region: awsRegionInput.value
        };
        localStorage.setItem('amazonApiKey', JSON.stringify(awsCreds));
    });
});

llmApiKeyInput.addEventListener('blur', () => {
    const provider = llmProviderSelect.value;
    const apiKey = llmApiKeyInput.value;
    if (apiKey && !['Amazon', 'Ollama'].includes(provider)) {
        localStorage.setItem(`${provider.toLowerCase()}ApiKey`, apiKey);
    }
});

ollamaHostInput.addEventListener('blur', () => {
    localStorage.setItem('ollamaHost', ollamaHostInput.value);
});


refreshModelsButton.addEventListener('click', fetchModels);

function updatePromptEditorState() {
    const hasChanged = promptEditorTextarea.value.trim() !== promptEditorTextarea.dataset.initialValue.trim();
    promptEditorSave.disabled = !hasChanged;

    let statusText = isPromptCustomForModel(currentProvider, currentModel) ? 'Custom' : 'Default';
    let statusClass = 'text-sm text-gray-400';

    if (hasChanged) {
        statusText = 'Unsaved Changes';
        statusClass = 'text-sm text-yellow-400';
    }

    promptEditorStatus.textContent = statusText;
    promptEditorStatus.className = statusClass;
}

function openPromptEditor() {
    promptEditorTitle.innerHTML = `System Prompt Editor for: <code class="text-teradata-orange font-normal">${currentProvider} / ${getNormalizedModelId(currentModel)}</code>`;
    const promptText = getSystemPromptForModel(currentProvider, currentModel);
    promptEditorTextarea.value = promptText;
    promptEditorTextarea.dataset.initialValue = promptText;

    promptEditorOverlay.classList.remove('hidden', 'opacity-0');
    promptEditorContent.classList.remove('scale-95', 'opacity-0');
    updatePromptEditorState();
}

function forceClosePromptEditor() {
    promptEditorOverlay.classList.add('opacity-0');
    promptEditorContent.classList.add('scale-95', 'opacity-0');
    setTimeout(() => {
        promptEditorOverlay.classList.add('hidden');
        promptEditorStatus.textContent = '';
    }, 300);
}

function closePromptEditor() {
    const hasChanged = promptEditorTextarea.value.trim() !== promptEditorTextarea.dataset.initialValue.trim();
    if (hasChanged) {
        showConfirmation(
            'Discard Changes?',
            'You have unsaved changes that will be lost. Are you sure you want to close the editor?',
            forceClosePromptEditor
        );
    } else {
        forceClosePromptEditor();
    }
}

async function saveSystemPromptChanges() {
    const newPromptText = promptEditorTextarea.value;
    const defaultPromptText = await getDefaultSystemPrompt(currentProvider, currentModel);

    if (defaultPromptText === null) {
        return;
    }

    const isCustom = newPromptText.trim() !== defaultPromptText.trim();

    saveSystemPromptForModel(currentProvider, currentModel, newPromptText, isCustom);
    updateStatusPromptName();

    promptEditorTextarea.dataset.initialValue = newPromptText;

    promptEditorStatus.textContent = 'Saved!';
    promptEditorStatus.className = 'text-sm text-green-400';
    setTimeout(() => {
        updatePromptEditorState();
    }, 2000);
}

async function resetSystemPrompt(force = false) {
    const defaultPrompt = await getDefaultSystemPrompt(currentProvider, currentModel);
    if (defaultPrompt) {
        if (!force) {
            promptEditorTextarea.value = defaultPrompt;
            updatePromptEditorState();
        } else {
            saveSystemPromptForModel(currentProvider, currentModel, defaultPrompt, false);
            promptEditorTextarea.value = defaultPrompt;
            updateStatusPromptName();
        }
    }
}

function showConfirmation(title, body, onConfirm) {
    confirmModalTitle.textContent = title;
    confirmModalBody.textContent = body;

    confirmModalOverlay.classList.remove('hidden', 'opacity-0');
    confirmModalContent.classList.remove('scale-95', 'opacity-0');

    const confirmHandler = () => {
        onConfirm();
        closeConfirmation();
    };

    const cancelHandler = () => {
        closeConfirmation();
    };

    const closeConfirmation = () => {
        confirmModalOverlay.classList.add('opacity-0');
        confirmModalContent.classList.add('scale-95', 'opacity-0');
        setTimeout(() => confirmModalOverlay.classList.add('hidden'), 300);
        confirmModalConfirm.removeEventListener('click', confirmHandler);
        confirmModalCancel.removeEventListener('click', cancelHandler);
    };

    confirmModalConfirm.addEventListener('click', confirmHandler, { once: true });
    confirmModalCancel.addEventListener('click', cancelHandler, { once: true });
}

async function handleIntensityChange() {
     if (isPromptCustomForModel(currentProvider, currentModel)) {
        showConfirmation(
            'Reset System Prompt?',
            'Changing the charting intensity requires resetting the system prompt to a new default to include updated instructions. Your custom changes will be lost. Do you want to continue?',
            () => {
                resetSystemPrompt(true);
                configStatus.textContent = 'Charting intensity updated and system prompt was reset to default.';
                configStatus.className = 'text-sm text-yellow-400 text-center';
            }
        );
    } else {
        await resetSystemPrompt(true);
        configStatus.textContent = 'Charting intensity updated.';
        configStatus.className = 'text-sm text-green-400 text-center';
    }
}

chartingIntensitySelect.addEventListener('change', handleIntensityChange);

promptEditorButton.addEventListener('click', openPromptEditor);
promptEditorClose.addEventListener('click', closePromptEditor);
promptEditorSave.addEventListener('click', saveSystemPromptChanges);
promptEditorReset.addEventListener('click', resetSystemPrompt.bind(null, false));
promptEditorTextarea.addEventListener('input', updatePromptEditorState);

function openChatModal() {
    chatModalOverlay.classList.remove('hidden', 'opacity-0');
    chatModalContent.classList.remove('scale-95', 'opacity-0');
    chatModalInput.focus();
}

function closeChatModal() {
    chatModalOverlay.classList.add('opacity-0');
    chatModalContent.classList.add('scale-95', 'opacity-0');
    setTimeout(() => chatModalOverlay.classList.add('hidden'), 300);
}

function addMessageToModal(role, content) {
    const messageEl = document.createElement('div');
    messageEl.className = `p-3 rounded-lg max-w-xs ${role === 'user' ? 'bg-blue-600 self-end' : 'bg-gray-600 self-start'}`;
    messageEl.textContent = content;
    chatModalLog.appendChild(messageEl);
    chatModalLog.scrollTop = chatModalLog.scrollHeight;
}

chatModalButton.addEventListener('click', openChatModal);
chatModalClose.addEventListener('click', closeChatModal);
chatModalOverlay.addEventListener('click', (e) => {
    if (e.target === chatModalOverlay) closeChatModal();
});

chatModalForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = chatModalInput.value.trim();
    if (!message) return;

    addMessageToModal('user', message);
    simpleChatHistory.push({ role: 'user', content: message });
    chatModalInput.value = '';
    chatModalInput.disabled = true;

    try {
        const res = await fetch('/simple_chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                history: simpleChatHistory
            })
        });

        const data = await res.json();

        if (res.ok) {
            addMessageToModal('assistant', data.response);
            simpleChatHistory.push({ role: 'assistant', content: data.response });
        } else {
            throw new Error(data.error || 'An unknown error occurred.');
        }

    } catch (error) {
        addMessageToModal('assistant', `Error: ${error.message}`);
    } finally {
        chatModalInput.disabled = false;
        chatModalInput.focus();
    }
});

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch('/app-config');
        const appConfig = await res.json();

        const chartingIntensityContainer = document.getElementById('charting-intensity-container');
        if (!appConfig.charting_enabled) {
            chartingIntensityContainer.style.display = 'none';
        }

    } catch (e) {
        console.error("Could not fetch app config", e);
    }

    const savedMcpConfig = JSON.parse(localStorage.getItem('mcpConfig'));
    if (savedMcpConfig) {
        document.getElementById('mcp-host').value = savedMcpConfig.host || '127.0.0.1';
        document.getElementById('mcp-port').value = savedMcpConfig.port || '8001';
        document.getElementById('mcp-path').value = savedMcpConfig.path || '/mcp/';
    } else {
         document.getElementById('mcp-host').value = '127.0.0.1';
        document.getElementById('mcp-port').value = '8001';
        document.getElementById('mcp-path').value = '/mcp/';
    }

    const lastProvider = localStorage.getItem('lastSelectedProvider');
    if (lastProvider) {
        llmProviderSelect.value = lastProvider;
    }
    llmProviderSelect.dispatchEvent(new Event('change'));

    toggleHistoryCheckbox.checked = !sessionHistoryPanel.classList.contains('collapsed');
    setupPanelToggle(toggleHistoryButton, sessionHistoryPanel, toggleHistoryCheckbox, historyCollapseIcon, historyExpandIcon);

    toggleHeaderCheckbox.checked = !toolHeader.classList.contains('collapsed');
    setupPanelToggle(toggleHeaderButton, toolHeader, toggleHeaderCheckbox, headerCollapseIcon, headerExpandIcon);

    toggleStatusCheckbox.checked = !statusWindow.classList.contains('collapsed');
    setupPanelToggle(toggleStatusButton, statusWindow, toggleStatusCheckbox, statusCollapseIcon, statusExpandIcon);

    configMenuButton.click();

    if (llmApiKeyInput.value || awsAccessKeyIdInput.value || ollamaHostInput.value) {
        await fetchModels();
    }

    configForm.addEventListener('input', updateConfigButtonState);
});