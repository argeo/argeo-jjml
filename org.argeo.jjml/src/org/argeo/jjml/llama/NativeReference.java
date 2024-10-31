package org.argeo.jjml.llama;

import java.util.function.LongSupplier;

/** Holds a reference the pointer of the underlying native structure. */
@Deprecated
abstract class NativeReference implements LongSupplier {
	private Long pointer;

	public NativeReference(Long pointer) {
		this.pointer = pointer;
	}

	@Deprecated
	public NativeReference() {
	}

	abstract void doDestroy(long pointer);

	/*
	 * UTILITIES
	 */
	protected void checkNotInitialized() {
		if (pointer != null)
			throw new IllegalStateException("Already initialized, destroy it first.");
	}

	protected void checkInitialized() {
		if (pointer == null)
			throw new IllegalStateException("Not initialized.");
	}

	public void destroy() {
		checkInitialized();
		doDestroy(pointer);
		pointer = null;
	}

	@Deprecated
	long getPointer() {
		return pointer;
	}

	@Deprecated
	protected void setPointer(long pointer) {
		this.pointer = pointer;
	}

	@Override
	public long getAsLong() {
		return pointer;
	}

}
