package org.argeo.jjml.llm.util.models;

public class ChatMlFormatter extends AbstractInstructFormatter {
	@Override
	protected void appendSystemPart(StringBuilder sb, String content) {
		sb.append("<|im_start|>system\n");
		sb.append(content);
		sb.append("<|im_end|>\n");
	}

	@Override
	protected void appendUserPart(StringBuilder sb, String content) {
		sb.append("<|im_start|>user\n");
		sb.append(content);
		sb.append("<|im_end|>\n");
	}

	@Override
	protected void appendAssistantPart(StringBuilder sb, String content) {
		sb.append("<|im_start|>assistant\n");
		sb.append(content);
		sb.append("<|im_end|>\n");
	}

	@Override
	public void appendGenerationPrompt(StringBuilder sb) {
		sb.append("<|im_start|>assistant\n");
	}

}
