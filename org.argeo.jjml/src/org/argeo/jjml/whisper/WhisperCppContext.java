package org.argeo.jjml.whisper;

import java.nio.charset.Charset;
import java.nio.file.Path;
import java.util.function.LongSupplier;

import org.argeo.jjml.llm.LlamaCppNative;

public class WhisperCppContext implements LongSupplier, AutoCloseable {
	static {
		LlamaCppNative.ensureLibrariesLoaded();
		WhisperNative.ensureLibrariesLoaded();
	}

	private final long pointer;

	public WhisperCppContext(Path modelPath) {
		this.pointer = doInit(filePathToNative(modelPath), true, true);
	}

	private static native long doInit(byte[] modelPath, boolean useGpu, boolean flashAttention);

	private native void doDestroy();

	@Override
	public long getAsLong() {
		return pointer;
	}

	@Override
	public void close() throws Exception {
		doDestroy();
	}

	/** Path as bytes, based on the OS native encoding. */
	private static byte[] filePathToNative(Path path) {
		return path.toString().getBytes(Charset.forName(System.getProperty("sun.jnu.encoding", "UTF-8")));
	}

}
