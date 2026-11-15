package org.argeo.jjml.llm.util.models;

public class Ministral3Formatter extends AbstractInstructFormatter {
	@Override
	protected void appendSystemPart(StringBuilder sb, String content) {
		sb.append("<s>[SYSTEM_PROMPT]");
		sb.append(content);
		sb.append("[/SYSTEM_PROMPT]");
	}

	@Override
	protected void appendUserPart(StringBuilder sb, String content) {
		sb.append("[INST]");
		sb.append(content);
		sb.append("[/INST]");
	}

	@Override
	protected void appendAssistantPart(StringBuilder sb, String content) {
		sb.append(content);
		sb.append("</s>");
	}

	@Override
	protected void appendToolPart(StringBuilder sb, String content) {
		sb.append("[TOOL_RESULTS]");
		sb.append(content);
		sb.append("[/TOOL_RESULTS]");
	}
	
	
}
