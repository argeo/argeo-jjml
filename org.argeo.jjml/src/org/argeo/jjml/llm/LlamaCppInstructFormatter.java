package org.argeo.jjml.llm;

import java.util.Arrays;
import java.util.List;

public interface LlamaCppInstructFormatter {
	String formatChatMessages(List<LlamaCppChatMessage> messages);

	default String formatChatMessages(LlamaCppChatMessage... messages) {
		return formatChatMessages(Arrays.asList(messages));
	}

}
