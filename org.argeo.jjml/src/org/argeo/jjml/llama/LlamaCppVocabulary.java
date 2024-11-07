package org.argeo.jjml.llama;

import java.io.PrintStream;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.CharBuffer;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

public class LlamaCppVocabulary {
	private final LlamaCppModel model;

	public LlamaCppVocabulary(LlamaCppModel model) {
		this.model = model;
	}

	// Tokenization
	/**
	 * Tokenize a string encoded as a standard UTF-8 byte array. To use when it
	 * makes more sense to convert on the Java side.
	 * 
	 * @see #doDeTokenizeAsUtf8Array(int[], boolean, boolean)
	 */
	private native int[] doTokenizeUtf8BytesAsArray(long pointer, byte[] str, int offset, int length,
			boolean addSpecial, boolean parseSpecial);

	private native int[] doTokenizeUtf8AsArray(long pointer, ByteBuffer str, boolean addSpecial, boolean parseSpecial);

	private native void doTokenizeUtf8Bytes(long pointer, byte[] str, int offset, int length, IntBuffer tokens,
			boolean addSpecial, boolean parseSpecial);

	private native void doTokenizeUtf8(long pointer, ByteBuffer str, IntBuffer tokens, boolean addSpecial,
			boolean parseSpecial);

	/** De-tokenize as a string encoded in standard UTF-8. */
	private native byte[] doDeTokenizeArrayAsUtf8Bytes(long pointer, int[] tokens, int offset, int length,
			boolean removeSpecial, boolean unparseSpecial);

	private native byte[] doDeTokenizeAsUtf8Bytes(long pointer, IntBuffer tokens, boolean removeSpecial,
			boolean unparseSpecial);

	private native void doDeTokenizeArrayAsUtf8(long pointer, int[] tokens, int offset, int length, ByteBuffer str,
			boolean removeSpecial, boolean unparseSpecial);

	private native void doDeTokenizeAsUtf8(long pointer, IntBuffer tokens, ByteBuffer str, boolean removeSpecial,
			boolean unparseSpecial);

	/**
	 * Tokenize a Java {@link String}. Its UTF-16 representation will be used
	 * without copy on the native side, where it will be converted to UTF-8.
	 */
	native int[] doTokenizeStringAsArray(long pointer, String str, boolean addSpecial, boolean parseSpecial);

	private native void doTokenizeString(long pointer, String str, IntBuffer buf, boolean addSpecial,
			boolean parseSpecial);

	/** De-tokenize as a Java {@link String}. */
	native String doDeTokenizeArrayAsString(long pointer, int[] tokens, int offset, int length, boolean removeSpecial,
			boolean unparseSpecial);

	private native String doDeTokenizeAsString(long pointer, IntBuffer buf, boolean removeSpecial,
			boolean unparseSpecial);

	/*
	 * UTF-8
	 */
	public void tokenize(CharSequence str, IntBuffer tokens, boolean addSpecial, boolean parseSpecial) {
		CharBuffer chars = CharBuffer.wrap(str);
		ByteBuffer utf8Str = StandardCharsets.UTF_8.encode(chars);
		tokenizeUtf8(utf8Str, tokens, addSpecial, parseSpecial);
	}

	public void tokenizeUtf8(ByteBuffer str, IntBuffer tokens, boolean addSpecial, boolean parseSpecial) {
		checkInput(str);
		checkOutput(tokens);
		synchronized (tokens) {// we are writing into this buffer and changing its position
			// make sure position is 0
			ByteBuffer in = str.slice().limit(str.limit() - str.position());
			IntBuffer out = tokens.slice().limit(tokens.limit() - tokens.position());

			if (out.isDirect()) {
				if (in.isDirect()) {
					doTokenizeUtf8(model.getAsLong(), in, out, addSpecial, parseSpecial);
				} else if (in.hasArray()) {
					byte[] arr = in.array();
					doTokenizeUtf8Bytes(model.getAsLong(), arr, in.arrayOffset(), in.limit() - in.position(), out,
							addSpecial, parseSpecial);
				} else {// copy
					byte[] copy = new byte[in.limit() - in.position()];
					in.get(copy, in.position(), copy.length);
					doTokenizeUtf8Bytes(model.getAsLong(), copy, 0, copy.length, out, addSpecial, parseSpecial);
				}
				tokens.position(tokens.position() + out.position());
			} else {
				int[] tokenArr;
				if (in.isDirect()) {
					tokenArr = doTokenizeUtf8AsArray(model.getAsLong(), in, addSpecial, parseSpecial);
				} else if (in.hasArray()) {
					byte[] arr = in.array();
					tokenArr = doTokenizeUtf8BytesAsArray(model.getAsLong(), arr, in.arrayOffset(),
							in.limit() - in.position(), addSpecial, parseSpecial);
				} else {// copy
					byte[] copy = new byte[in.limit() - in.position()];
					in.get(copy, in.position(), copy.length);
					tokenArr = doTokenizeUtf8BytesAsArray(model.getAsLong(), copy, 0, copy.length, addSpecial,
							parseSpecial);
				}
				if (tokenArr.length > (tokens.limit() - tokens.position()))
					throw new IndexOutOfBoundsException(tokenArr.length);
				tokens.put(tokenArr);
			}
		}
	}

	public void deTokenize(IntBuffer in, ByteBuffer str, boolean removeSpecial, boolean unparseSpecial) {
		deTokenizeUtf8(in, str, removeSpecial, unparseSpecial);
	}

	public void deTokenizeUtf8(IntBuffer in, ByteBuffer str, boolean removeSpecial, boolean unparseSpecial) {
		doDeTokenizeAsUtf8(model.getAsLong(), in, str, removeSpecial, unparseSpecial);
	}

	/*
	 * UTF-16
	 */
	/**
	 * Tokenize, with the native side retrieving as efficiently as possible the
	 * UTF-16 Java internal string representation and performing the UTF-16->UTF-8
	 * conversion on its side.
	 * 
	 * If the input is a <code>String</code>, it is guaranteed that it will be used
	 * directly, since {@link CharSequence#toString()} is being used as input to the
	 * native side.
	 * 
	 * @param str the input characters
	 */
	public void tokenizeUtf16(CharSequence str, IntBuffer tokens, boolean addSpecial, boolean parseSpecial) {
		Objects.requireNonNull(str);
		checkOutput(tokens);
		synchronized (tokens) {// we are writing into this buffer and changing its position
			IntBuffer tokensToUse = tokens.slice().limit(tokens.limit() - tokens.position());
			String in = str.toString();
			assert str instanceof String ? in == str : true;
			if (tokens.isDirect()) {
				doTokenizeString(model.getAsLong(), in, tokensToUse, addSpecial, parseSpecial);
				tokens.position(tokens.position() + tokensToUse.position());
			} else {
				int[] tokenArr = doTokenizeStringAsArray(model.getAsLong(), in, addSpecial, parseSpecial);
				if (tokenArr.length > (tokens.limit() - tokens.position()))
					throw new IndexOutOfBoundsException(tokenArr.length);
				tokens.put(tokenArr);
			}
		}
	}

	public String deTokenizeUtf16(IntBuffer in, boolean removeSpecial, boolean unparseSpecial) {
		Objects.requireNonNull(in);
		if (in.isDirect()) {
			return doDeTokenizeAsString(model.getAsLong(), in, removeSpecial, unparseSpecial);
		} else {
			return doDeTokenizeArrayAsString(model.getAsLong(), in.array(), in.position(), in.limit() - in.position(),
					removeSpecial, unparseSpecial);
		}
	}

	/*
	 * UTILITIES
	 */
	private void checkInput(Buffer in) {
		if (in instanceof IntBuffer buf && !ByteOrder.nativeOrder().equals(buf.order()))
			throw new IllegalArgumentException("Int buffer does not use native byte order");
		Objects.requireNonNull(in, "Input buffer cannot be null");
	}

	private void checkOutput(Buffer out) {
		Objects.requireNonNull(out, "Output buffer cannot be null");
		if (out.isReadOnly())
			throw new IllegalArgumentException("Output buffer is read-only");
		if (out instanceof IntBuffer buf && !ByteOrder.nativeOrder().equals(buf.order()))
			throw new IllegalArgumentException("Int buffer does not use native byte order");
	}

	/*
	 * STATIC UTILITIES
	 */
	/**
	 * Write the beginning of an integer buffer as a string. It has no side effect
	 * on the input buffer.
	 */
	static String logIntegers(IntBuffer in, int max, String separator) {
		StringBuilder sb = new StringBuilder();
		integers: for (int i = in.position(); i < in.limit(); i++) {
			if (i != in.position())
				sb.append(separator);
			if (i == max) {
				sb.append("...");
				break integers;
			}
			sb.append(Integer.toString(in.get(i)));
		}
		return sb.toString();
	}

}
