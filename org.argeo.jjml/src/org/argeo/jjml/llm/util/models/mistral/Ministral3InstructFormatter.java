package org.argeo.jjml.llm.util.models.mistral;

import java.util.List;

import org.argeo.jjml.llm.LlamaCppChatMessage;
import org.argeo.jjml.llm.util.InstructRole;
import org.argeo.jjml.llm.util.models.AbstractInstructFormatter;

public class Ministral3InstructFormatter extends AbstractInstructFormatter {
	@Override
	public String formatChatMessages(List<LlamaCppChatMessage> messages) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < messages.size(); i++) {
			LlamaCppChatMessage msg = messages.get(i);
			String roleStr = msg.getRole();
			InstructRole role = InstructRole.valueOf(roleStr.toUpperCase());
			switch (role) {
			case SYSTEM:
				appendSystemPart(sb, msg.getContent());
				break;
			case USER:
				appendSystemPart(sb, msg.getContent());
				break;
			case ASSISTANT:
				appendSystemPart(sb, msg.getContent());
				break;

			default:
				break;
			}
		}
		return sb.toString();
	}

	protected void appendSystemPart(StringBuilder sb, String content) {
		sb.append("[SYSTEM_PROMPT]");
		sb.append(content);
		sb.append("[/SYSTEM_PROMPT]");
	}

	protected void appendUserPart(StringBuilder sb, String content) {
		sb.append("[INST]");
		sb.append(content);
		sb.append("[/INST]");
	}

	protected void appendAssistantPart(StringBuilder sb, String content) {
		sb.append(content);
		sb.append("</s>");
	}
}
