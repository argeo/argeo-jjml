package org.argeo.jjml.llama;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

public class LlamaCppModel {
	private Path localPath;

	// implementation
	private Long pointer = null;

	/*
	 * NATIVE METHODS
	 */

	native int[] doTokenize(String str, boolean addSpecial, boolean parseSpecial);

	native String doDeTokenize(int[] tokens, boolean special);

	native long doInit(String localPathStr);

	native void doDestroy();

	native String doFormatChatMessages(String[] roles, String[] contents);

	/*
	 * USABLE METHODS
	 */
	public String deTokenize(LlamaCppTokenList tokenList, boolean special) {
		return doDeTokenize(tokenList.getTokens(), special);
	}

	public LlamaCppTokenList tokenize(String str, boolean addSpecial) {
		return tokenize(str, addSpecial, false);
	}

	public LlamaCppTokenList tokenize(String str, boolean addSpecial, boolean parseSpecial) {
		int[] tokens = doTokenize(str, addSpecial, parseSpecial);
		return new LlamaCppTokenList(this, tokens);
	}

	public String formatChatMessages(List<LlamaCppChatMessage> messages) {
		String[] roles = new String[messages.size()];
		String[] contents = new String[messages.size()];

		for (int i = 0; i < messages.size(); i++) {
			LlamaCppChatMessage message = messages.get(i);
			roles[i] = message.role();
			contents[i] = message.content();
		}

		String res = doFormatChatMessages(roles, contents);
		return res;
	}

	/*
	 * LIFECYCLE
	 */

	public void init() {
		if (pointer != null)
			throw new IllegalStateException("Model is already initialized.");
		Objects.requireNonNull(localPath, "Local path to the model must be set");
		if (!Files.exists(localPath))
			throw new IllegalArgumentException("Model file does not exist: " + localPath);
		pointer = doInit(localPath.toString());
	}

	public void destroy() {
		Objects.requireNonNull(pointer, "Model must be initialized");
		doDestroy();
	}

	/*
	 * ACESSORS
	 */

	public Path getLocalPath() {
		return localPath;
	}

//	String getLocalPathAsString() {
//		Objects.requireNonNull(localPath);
//		return localPath.toString();
//	}

	public void setLocalPath(Path localPath) {
		this.localPath = localPath;
	}

	final long getPointer() {
		return pointer;
	}

}
