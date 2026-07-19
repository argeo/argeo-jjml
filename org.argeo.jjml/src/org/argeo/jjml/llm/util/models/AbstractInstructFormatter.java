package org.argeo.jjml.llm.util.models;

import java.util.List;

import org.argeo.jjml.llm.LlamaCppChatMessage;
import org.argeo.jjml.llm.LlamaCppInstructFormatter;
import org.argeo.jjml.llm.util.InstructRole;

public abstract class AbstractInstructFormatter implements LlamaCppInstructFormatter {

	protected abstract void appendSystemPart(StringBuilder sb, String content);

	protected abstract void appendUserPart(StringBuilder sb, String content);

	protected abstract void appendAssistantPart(StringBuilder sb, String content);

	@Override
	public String formatChatMessages(List<LlamaCppChatMessage> messages) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < messages.size(); i++) {
			LlamaCppChatMessage msg = messages.get(i);
			String roleStr = msg.getRole();
			InstructRole role = InstructRole.valueOf(roleStr.toUpperCase());
			switch (role) {
			case SYSTEM:
				if (i != 0)
					throw new IllegalStateException("System prompt must be first");
				appendSystemPart(sb, msg.getContent());
				break;
			case USER:
				appendUserPart(sb, msg.getContent());
				break;
			case ASSISTANT:
				appendAssistantPart(sb, msg.getContent());
				break;

			default:
				break;
			}
		}
		return sb.toString();
	}

}
