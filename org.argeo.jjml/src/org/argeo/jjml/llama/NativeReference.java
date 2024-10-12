package org.argeo.jjml.llama;

/** Holds a reference the pointer of the underlying native structure. */
abstract class NativeReference {
	private Long pointer;

	abstract void doDestroy();

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
		doDestroy();
		pointer = null;
	}

	long getPointer() {
		return pointer;
	}

	protected void setPointer(long pointer) {
		this.pointer = pointer;
	}

}
