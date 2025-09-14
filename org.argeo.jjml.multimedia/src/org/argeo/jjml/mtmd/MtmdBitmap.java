package org.argeo.jjml.mtmd;

import java.util.function.LongSupplier;

public abstract class MtmdBitmap implements LongSupplier, AutoCloseable {
	private final long pointer;

	protected MtmdBitmap(long pointer) {
		this.pointer = pointer;
	}

	public abstract MtmdInputChunkType getType();

	private native void doDestroy();

	@Override
	public long getAsLong() {
		return pointer;
	}

	@Override
	public void close() throws Exception {
		doDestroy();
	}

}
