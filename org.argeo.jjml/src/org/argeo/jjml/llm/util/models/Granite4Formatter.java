package org.argeo.jjml.llm.util.models;

public class Granite4Formatter extends AbstractInstructFormatter {

	@Override
	protected void appendSystemPart(StringBuilder sb, String content) {
		sb.append("<|start_of_role|>system<|end_of_role|>");
		sb.append(content);
		sb.append("<|end_of_text|>\n");
	}

	@Override
	protected void appendUserPart(StringBuilder sb, String content) {
		sb.append("<|start_of_role|>user<|end_of_role|>");
		sb.append(content);
		sb.append("<|end_of_text|>\n");
	}

	@Override
	protected void appendAssistantPart(StringBuilder sb, String content) {
		sb.append("<|start_of_role|>assistant<|end_of_role|>");
		sb.append(content);
		// sb.append("<|end_of_text|>"); // EOG? EOS?
	}

	@Override
	public void appendGenerationPrompt(StringBuilder sb) {
		sb.append("<|start_of_role|>assistant<|end_of_role|>");
	}

}
