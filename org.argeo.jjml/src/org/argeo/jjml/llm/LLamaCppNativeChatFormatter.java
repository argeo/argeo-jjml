package org.argeo.jjml.llm;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.util.List;
import java.util.function.Predicate;

public class LLamaCppNativeChatFormatter {

	/*
	 * NATIVE METHODS
	 */
	private static native byte[] doFormatChatMessages(byte[][] utf8Roles, byte[][] utf8Contents,
			boolean addAssistantTokens, byte[] ut8ChatTemplate);

	/*
	 * USABLE METHODS
	 */
	static String formatChatMessages(List<LlamaCppChatMessage> messages,
			Predicate<LlamaCppChatMessage> addAssistantTokens, String chatTemplate) {
		byte[][] roles = new byte[messages.size()][];
		byte[][] contents = new byte[messages.size()][];

		boolean currIsUserRole = false;
		for (int i = 0; i < messages.size(); i++) {
			LlamaCppChatMessage message = messages.get(i);
			roles[i] = message.getRole().getBytes(UTF_8);
			currIsUserRole = addAssistantTokens.test(message);
			contents[i] = message.getContent().getBytes(UTF_8);
		}

		byte[] res = doFormatChatMessages(roles, contents, currIsUserRole, chatTemplate.getBytes(UTF_8));
		return new String(res, UTF_8);
	}

}
