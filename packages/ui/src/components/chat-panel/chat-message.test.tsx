
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ChatMessage } from './chat-message';
import type { ChatMessageEntry } from './chat-message';

describe('ChatMessage', () => {
    it('should render user message on the right', () => {
        const msg: ChatMessageEntry = { id: '1', role: 'user', content: 'Hello there' };
        render(<ChatMessage message={msg} />);

        expect(screen.getByText('Hello there')).toBeInTheDocument();
    });

    it('should render assistant message with bot icon', () => {
        const msg: ChatMessageEntry = { id: '2', role: 'assistant', content: 'Here is the answer.' };
        render(<ChatMessage message={msg} />);

        expect(screen.getByText('Here is the answer.')).toBeInTheDocument();
    });

    it('should render loading state with spinner text', () => {
        const msg: ChatMessageEntry = { id: '3', role: 'assistant', content: '', isLoading: true };
        render(<ChatMessage message={msg} />);

        expect(screen.getByText('Thinking...')).toBeInTheDocument();
    });

    it('should render error state', () => {
        const msg: ChatMessageEntry = { id: '4', role: 'assistant', content: '', error: 'Model timeout' };
        render(<ChatMessage message={msg} />);

        expect(screen.getByText('Model timeout')).toBeInTheDocument();
    });

    it('should render low confidence warning', () => {
        const msg: ChatMessageEntry = { id: '5', role: 'assistant', content: 'Maybe...', lowConfidence: true };
        render(<ChatMessage message={msg} />);

        expect(screen.getByText(/low confidence/i)).toBeInTheDocument();
        expect(screen.getByText('Maybe...')).toBeInTheDocument();
    });
});
