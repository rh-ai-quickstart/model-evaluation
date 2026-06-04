
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ChatInput } from './chat-input';

describe('ChatInput', () => {
    it('should call onSend with trimmed text when clicking send', () => {
        const onSend = vi.fn();
        render(<ChatInput onSend={onSend} />);

        const textarea = screen.getByPlaceholderText(/ask a compliance question/i);
        fireEvent.change(textarea, { target: { value: '  What are the rules?  ' } });
        fireEvent.click(screen.getByRole('button'));

        expect(onSend).toHaveBeenCalledWith('What are the rules?');
    });

    it('should submit on Enter and clear input', () => {
        const onSend = vi.fn();
        render(<ChatInput onSend={onSend} />);

        const textarea = screen.getByPlaceholderText(/ask a compliance question/i);
        fireEvent.change(textarea, { target: { value: 'Test question' } });
        fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

        expect(onSend).toHaveBeenCalledWith('Test question');
    });

    it('should not submit on Shift+Enter', () => {
        const onSend = vi.fn();
        render(<ChatInput onSend={onSend} />);

        const textarea = screen.getByPlaceholderText(/ask a compliance question/i);
        fireEvent.change(textarea, { target: { value: 'line one' } });
        fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });

        expect(onSend).not.toHaveBeenCalled();
    });

    it('should not submit empty input', () => {
        const onSend = vi.fn();
        render(<ChatInput onSend={onSend} />);

        fireEvent.click(screen.getByRole('button'));

        expect(onSend).not.toHaveBeenCalled();
    });

    it('should disable textarea and button when disabled', () => {
        const onSend = vi.fn();
        render(<ChatInput onSend={onSend} disabled />);

        expect(screen.getByPlaceholderText(/ask a compliance question/i)).toBeDisabled();
        expect(screen.getByRole('button')).toBeDisabled();
    });
});
