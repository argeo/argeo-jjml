package org.argeo.jjml.mtmd;

import java.nio.charset.Charset;
import java.nio.file.Path;
import java.util.function.LongSupplier;

import org.argeo.jjml.llm.LlamaCppModel;

public class MtmdContext implements LongSupplier, AutoCloseable {
	private final long pointer;

	public MtmdContext(LlamaCppModel model, Path mmprojPath, int threads) {
		this.pointer = doInit(model, filePathToNative(mmprojPath), true, threads);
	}

	private static native long doInit(LlamaCppModel model, byte[] mmprojPath, boolean useGpu, int threads);

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
