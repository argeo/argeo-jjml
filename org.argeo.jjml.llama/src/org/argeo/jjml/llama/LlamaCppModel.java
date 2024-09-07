package org.argeo.jjml.llama;

import java.nio.file.Path;
import java.util.Objects;

public class LlamaCppModel {
	private Path localPath;

	native void doInit();

	native void doDestroy();

	public void init() {
		Objects.requireNonNull(localPath, "Local path to the model must be set");
		doInit();
	}

	public void destroy() {
		doDestroy();
	}

	public Path getLocalPath() {
		return localPath;
	}

	public void setLocalPath(Path localPath) {
		this.localPath = localPath;
	}

}
