package org.argeo.jjml.llm.util.models;

import org.argeo.jjml.llm.instruct.LlamaCppInstructFormatter;
import org.argeo.jjml.llm.instruct.LlamaCppInstructPart;
import org.argeo.jjml.llm.util.InstructRole;

public abstract class AbstractInstructFormatter implements LlamaCppInstructFormatter {

	protected abstract void appendSystemPart(StringBuilder sb, String content);

	protected abstract void appendUserPart(StringBuilder sb, String content);

	protected abstract void appendAssistantPart(StringBuilder sb, String content);

	protected void appendToolPart(StringBuilder sb, String content) {
		throw new UnsupportedOperationException("Tool messages are not supported.");
	}

	@Override
	public String formatMessages(Iterable<? extends LlamaCppInstructPart> messages) {
		StringBuilder sb = new StringBuilder();
		int index = 0;
		for (LlamaCppInstructPart msg : messages) {
			String roleStr = msg.getRole();
			InstructRole role = InstructRole.valueOf(roleStr.toUpperCase());
			switch (role) {
			case SYSTEM:
				if (index != 0)
					throw new IllegalStateException("System prompt must be first");
				appendSystemPart(sb, msg.toString());
				break;
			case USER:
				appendUserPart(sb, msg.toString());
				break;
			case ASSISTANT:
				appendAssistantPart(sb, msg.toString());
				break;
			case TOOL:
				appendToolPart(sb, msg.toString());
				break;

			default:
				break;
			}
			index++;
		}
		return sb.toString();
	}

}
