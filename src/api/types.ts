export interface CompletionChunk {
    choices: CompletionChoice[];
    usage?: {
        completion_tokens: number;
        prompt_tokens: number;
        total_tokens: number;
    };
}

export interface CompletionChoice {
    finish_reason?: string;
    index: number;
    delta?: ChatMessage;
    message?: ChatMessage;
}

export interface ChatMessage {
    role?: string;
    content?: string;
    reasoning_content?: string;
    tool_calls?: ToolCall[];
}

export interface ChatRequestMessage {
    role: string;
    content?: string;
    reasoning_content?: string;
    tool_calls?: ToolCall[];
}

export interface CompletionRequest {
    model?: string;
    stream?: boolean;
    messages: ChatRequestMessage[];
    max_tokens?: number;
    continue_final_message?: boolean;
    add_generation_prompt?: boolean;
    [key: string]: unknown;
}

export interface ToolCall {
    index?: number;
    id?: string;
    type?: string;
    function?: {
        name?: string;
        arguments?: string;
    };
}