package com.example.testjjml;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import org.argeo.jjml.llm.LlamaCppInstructProcessor;
import org.argeo.jjml.llm.util.InstructRole;

import java.io.Writer;

public class ChatFragment extends Fragment implements LlmManager.Listener {

    private LinearLayout chatHistory;
    private ScrollView scrollView;
    private EditText chatInput;
    private Button sendButton;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_chat, container, false);

        chatHistory = view.findViewById(R.id.chat_history);
        scrollView = view.findViewById(R.id.chat_scroll_view);
        chatInput = view.findViewById(R.id.chat_input);
        sendButton = view.findViewById(R.id.send_button);

        sendButton.setOnClickListener(v -> sendMessage());

        LlmManager.INSTANCE.setListener(this);

        if (LlmManager.INSTANCE.isLoaded()) {
            onLoaded();
        } else {
            addChatBubble("System", LlmManager.INSTANCE.getStatus());
        }

        return view;
    }

    private void sendMessage() {
        String text = chatInput.getText().toString().trim();
        if (text.isEmpty()) return;

        chatInput.setText("");
        sendButton.setEnabled(false);

        addChatBubble("User", text);
        TextView assistantBubble = addChatBubble("Assistant", "...");

        LlamaCppInstructProcessor processor = LlmManager.INSTANCE.getProcessor();
        if (processor == null) {
            assistantBubble.setText("Error: Model not loaded.");
            sendButton.setEnabled(true);
            return;
        }

        LlmManager.INSTANCE.executeInBg(() -> {
            try {
                processor.write(InstructRole.USER, text);

                UiWriter writer = new UiWriter(assistantBubble);
                processor.readMessage(writer);

                new Handler(Looper.getMainLooper()).post(() -> sendButton.setEnabled(true));
            } catch (Exception e) {
                e.printStackTrace();
                new Handler(Looper.getMainLooper()).post(() -> {
                    assistantBubble.setText(assistantBubble.getText() + "\n[Error: " + e.getMessage() + "]");
                    sendButton.setEnabled(true);
                });
            }
        });
    }

    private TextView addChatBubble(String role, String text) {
        View bubble = getLayoutInflater().inflate(R.layout.item_chat_bubble, chatHistory, false);
        TextView roleTv = bubble.findViewById(R.id.chat_role);
        TextView textTv = bubble.findViewById(R.id.chat_text);

        roleTv.setText(role);
        textTv.setText(text);

        chatHistory.addView(bubble);

        // Scroll to bottom
        scrollView.post(() -> scrollView.fullScroll(View.FOCUS_DOWN));

        return textTv;
    }

    @Override
    public void onStatusUpdate(String status) {
        if (getContext() == null) return;
        addChatBubble("System", status);
    }

    @Override
    public void onLoaded() {
        if (getContext() == null) return;
        addChatBubble("System", "Model is ready to chat!");
        sendButton.setEnabled(true);

        // Initialize the processor with system prompt if this is the first time
        LlmManager.INSTANCE.executeInBg(() -> {
            try {
                LlamaCppInstructProcessor processor = LlmManager.INSTANCE.getProcessor();
                // Usually we write the system prompt once, but for simplicity here we just do it once.
                // processor.write(InstructRole.SYSTEM, "You are a helpful AI assistant.");
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        LlmManager.INSTANCE.setListener(null);
    }

    private class UiWriter extends Writer {
        private final TextView targetView;
        private final StringBuilder content = new StringBuilder();

        public UiWriter(TextView targetView) {
            this.targetView = targetView;
        }

        @Override
        public void write(char[] cbuf, int off, int len) {
            String chunk = new String(cbuf, off, len);
            content.append(chunk);

            new Handler(Looper.getMainLooper()).post(() -> {
                targetView.setText(content.toString());
                scrollView.post(() -> scrollView.fullScroll(View.FOCUS_DOWN));
            });
        }

        @Override
        public void flush() {}

        @Override
        public void close() {}
    }
}
