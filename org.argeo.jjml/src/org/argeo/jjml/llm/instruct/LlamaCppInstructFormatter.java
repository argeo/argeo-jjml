package org.argeo.jjml.llm.instruct;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.argeo.jjml.llm.LlamaCppChatMessage;

public interface LlamaCppInstructFormatter {
	String formatMessages(Iterable<? extends LlamaCppInstructPart> messages);

	default String formatMessage(LlamaCppInstructPart message) {
		return formatMessages(Collections.singleton(message));
	}

	default void appendGenerationPrompt(StringBuilder sb) {
	}

	@Deprecated
	default String formatChatMessages(List<LlamaCppChatMessage> messages) {
		return formatMessages(messages);
	}

	@Deprecated
	default String formatChatMessages(LlamaCppChatMessage... messages) {
		return formatChatMessages(Arrays.asList(messages));
	}
}
