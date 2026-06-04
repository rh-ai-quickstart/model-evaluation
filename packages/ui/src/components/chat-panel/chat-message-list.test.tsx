
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ChatMessageList } from './chat-message-list';

describe('ChatMessageList', () => {
    it('should render welcome state when no messages', () => {
        render(<ChatMessageList messages={[]} modelName="granite-8b" onSuggestedClick={vi.fn()} />);

        expect(screen.getByText(/FSI Compliance Assistant/i)).toBeInTheDocument();
        expect(screen.getByText(/granite-8b/i)).toBeInTheDocument();
    });

    it('should render suggested questions as clickable buttons', () => {
        const onSuggestedClick = vi.fn();
        render(<ChatMessageList messages={[]} modelName="granite-8b" onSuggestedClick={onSuggestedClick} />);

        const buttons = screen.getAllByRole('button');
        expect(buttons.length).toBe(3);

        fireEvent.click(buttons[0]);
        expect(onSuggestedClick).toHaveBeenCalledOnce();
    });

    it('should disable suggested questions when no model selected', () => {
        render(<ChatMessageList messages={[]} modelName={null} onSuggestedClick={vi.fn()} />);

        const buttons = screen.getAllByRole('button');
        buttons.forEach((btn) => expect(btn).toBeDisabled());
    });

    it('should render messages instead of welcome state', () => {
        const messages = [
            { id: '1', role: 'user' as const, content: 'Hello' },
            { id: '2', role: 'assistant' as const, content: 'Hi there' },
        ];
        render(<ChatMessageList messages={messages} modelName="granite-8b" onSuggestedClick={vi.fn()} />);

        expect(screen.queryByText(/FSI Compliance Assistant/i)).not.toBeInTheDocument();
    });
});
