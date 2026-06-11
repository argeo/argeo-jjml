package org.argeo.jjml.llm;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.argeo.jjml.llm.util.InstructRole;

/**
 * Format chat messages using llama.cpp basic capabilities (Jinja templates are
 * <b>not</b> supported).
 */
public class LLamaCppNativeChatFormatter implements LlamaCppInstructFormatter {

	private String chatTemplate;

	public LLamaCppNativeChatFormatter(String chatTemplate) {
		this.chatTemplate = chatTemplate;
	}

	/*
	 * NATIVE METHODS
	 */
	private static native byte[] doFormatChatMessages(byte[][] utf8Roles, byte[][] utf8Contents,
			boolean addAssistantTokens, byte[] ut8ChatTemplate);

	/*
	 * LlamaCppInstructFormatter IMPLEMENTATION
	 */
	@Override
	public String formatChatMessages(List<LlamaCppChatMessage> messages) {
		String formatted = LLamaCppNativeChatFormatter.formatChatMessages(messages, //
				(message) -> message.getRole().equals(InstructRole.USER.get()), chatTemplate);
		return formatted;
	}

	/*
	 * USABLE METHODS
	 */
	/**
	 * Format a list of chat messages either as 'user' or 'assistant' messages.
	 * 
	 * @param messages           the list of qualified chat messages
	 * @param addAssistantTokens whether a given message should be considered 'user'
	 *                           (returns <code>true</code>) or 'assistant'
	 * @param chatTemplate       the llama.cpp id for the chat template (e.g.
	 *                           'granite'), not a full template
	 * @return the formatted messages as single string
	 */
	static String formatChatMessages(List<LlamaCppChatMessage> messages,
			Predicate<LlamaCppChatMessage> addAssistantTokens, String chatTemplate) {
		// filter out null values
		List<LlamaCppChatMessage> msgs = messages.stream().filter(Objects::nonNull).collect(Collectors.toList());
		byte[][] roles = new byte[msgs.size()][];
		byte[][] contents = new byte[msgs.size()][];

		boolean currIsUserRole = false;
		messages: for (int i = 0; i < msgs.size(); i++) {
			LlamaCppChatMessage message = msgs.get(i);
			if (message == null)
				continue messages; // ignore
			roles[i] = message.getRole().getBytes(UTF_8);
			currIsUserRole = addAssistantTokens.test(message);
			contents[i] = message.getContent().getBytes(UTF_8);
		}

		byte[] res = doFormatChatMessages(roles, contents, currIsUserRole, chatTemplate.getBytes(UTF_8));
		return new String(res, UTF_8);
	}

}
