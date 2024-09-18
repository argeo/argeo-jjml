package org.argeo.jjml.llama;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

public class LlamaCppModel {
	private Path localPath;

	// implementation
	private Long pointer = null;

	native int[] doTokenize(String str, boolean addSpecial, boolean parseSpecial);

	native String doDeTokenize(int[] tokens, boolean special);

	native long doInit(String localPathStr);

	native void doDestroy();

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
