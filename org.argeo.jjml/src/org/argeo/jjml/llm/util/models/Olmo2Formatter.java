package org.argeo.jjml.llm.util.models;

public class Olmo2Formatter extends AbstractInstructFormatter {
	@Override
	protected void appendSystemPart(StringBuilder sb, String content) {
		sb.append("<|system|>\n");
		sb.append(content);
		sb.append("\n");
	}

	@Override
	protected void appendUserPart(StringBuilder sb, String content) {
		sb.append("<|user|>\n");
		sb.append(content);
		sb.append("\n");
	}

	@Override
	protected void appendAssistantPart(StringBuilder sb, String content) {
		sb.append("<|assistant|>\n");
		sb.append(content);
		sb.append("\n");
	}

	@Override
	public void appendGenerationPrompt(StringBuilder sb) {
		sb.append("<|assistant|>\n");
	}

}
