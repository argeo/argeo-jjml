package org.argeo.jjml.llama;

import java.nio.file.Path;
import java.util.Objects;

public class LlamaCppModel {
	private Path localPath;

	private Long pointer = null;

	native long doInit(String localPathStr);

	native void doDestroy(long pointer);

	public void init() {
		if (pointer != null)
			throw new IllegalStateException("Model is already initialized.");
		Objects.requireNonNull(localPath, "Local path to the model must be set");
		pointer = doInit(localPath.toString());
	}

	public void destroy() {
		Objects.requireNonNull(pointer, "Model must be initialized");
		doDestroy(pointer);
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

	Long getPointer() {
		return pointer;
	}

}
