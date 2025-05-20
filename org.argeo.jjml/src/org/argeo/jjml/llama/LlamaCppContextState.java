package org.argeo.jjml.llama;

import java.nio.ByteBuffer;

public interface LlamaCppContextState {
	void save(LlamaCppContext context, int contextPosition);

	int load(LlamaCppContext context);

	static class ByteBufferSavedState implements LlamaCppContextState {
		private ByteBuffer savedState;
		private int savedContextPosition;

		@Override
		public void save(LlamaCppContext context, int contextPosition) {
			int stateSize = (int) context.getStateSize();
			savedState = ByteBuffer.allocate(stateSize);
			context.readState(savedState);
			System.out.println("Saved context state (" + stateSize / (1024 * 1024) + " MiB)");
			savedContextPosition = contextPosition;

		}

		@Override
		public int load(LlamaCppContext context) {
			context.writeState(savedState);
			return savedContextPosition;
		}

	}
}
