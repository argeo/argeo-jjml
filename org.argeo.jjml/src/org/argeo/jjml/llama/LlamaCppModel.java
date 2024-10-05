package org.argeo.jjml.llama;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.function.DoublePredicate;

public class LlamaCppModel {
	private Path localPath;

	// implementation
	private Long pointer = null;

	private LlamaCppModelParams initParams;

	private int embeddingSize;

	public LlamaCppModel() {
		LlamaCppBackend.ensureInitialized();
	}

	/*
	 * NATIVE METHODS
	 */

	native int[] doTokenize(String str, boolean addSpecial, boolean parseSpecial);

	native String doDeTokenize(int[] tokens, boolean special);

	native long doInit(String localPathStr, LlamaCppModelParams params, DoublePredicate progressCallback);

	native void doDestroy();

	native String doFormatChatMessages(String[] roles, String[] contents);

	native int doGetEmbeddingSize();

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
	/** Init model with defaults. */
	public void init() {
		init(LlamaCppModelParams.defaultModelParams(), null);
	}

	public void init(LlamaCppModelParams modelParams, DoublePredicate progressCallback) {
		if (pointer != null)
			throw new IllegalStateException("Model is already initialized.");
		Objects.requireNonNull(localPath, "Local path to the model must be set");
		if (!Files.exists(localPath))
			throw new IllegalArgumentException("Model file does not exist: " + localPath);
		pointer = doInit(localPath.toString(), modelParams, progressCallback);
		this.initParams = modelParams;
		embeddingSize = doGetEmbeddingSize();
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

	public LlamaCppModelParams getInitParams() {
		return initParams;
	}

	public int getEmbeddingSize() {
		return embeddingSize;
	}

	final long getPointer() {
		return pointer;
	}

}
