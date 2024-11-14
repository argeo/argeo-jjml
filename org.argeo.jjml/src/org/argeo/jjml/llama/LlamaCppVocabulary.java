package org.argeo.jjml.llama;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.CharBuffer;
import java.nio.IntBuffer;
import java.util.List;
import java.util.Objects;

public class LlamaCppVocabulary {
	/**
	 * Whether Java <-> UTF-8 conversion happens on the native side (true) or on the
	 * Java side (false).
	 */
	private boolean stringMode = false;

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
	private static native int[] doTokenizeUtf8BytesAsArray(long pointer, byte[] str, int offset, int length,
			boolean addSpecial, boolean parseSpecial);

	private static native int[] doTokenizeUtf8AsArray(long pointer, ByteBuffer str, int offset, int length,
			boolean addSpecial, boolean parseSpecial);

	private static native int doTokenizeUtf8(long pointer, ByteBuffer str, int offset, int length, IntBuffer tokens,
			int pos, int size, boolean addSpecial, boolean parseSpecial);

	/** De-tokenize as a string encoded in standard UTF-8. */
	private static native byte[] doDeTokenizeArrayAsUtf8Bytes(long pointer, int[] tokens, int pos, int size,
			boolean removeSpecial, boolean unparseSpecial);

	private static native byte[] doDeTokenizeAsUtf8Bytes(long pointer, IntBuffer tokens, int pos, int size,
			boolean removeSpecial, boolean unparseSpecial);

	private static native int doDeTokenizeAsUtf8(long pointer, IntBuffer tokens, int pos, int size, ByteBuffer str,
			int offset, int length, boolean removeSpecial, boolean unparseSpecial);

	/**
	 * Tokenize a Java {@link String}. Its UTF-16 representation will be used
	 * without copy on the native side, where it will be converted to UTF-8.
	 */
	private static native int[] doTokenizeStringAsArray(long pointer, String str, boolean addSpecial,
			boolean parseSpecial);

	/** De-tokenize as a Java {@link String}. */
	private static native String doDeTokenizeArrayAsString(long pointer, int[] tokens, int pos, int size,
			boolean removeSpecial, boolean unparseSpecial);

	private static native String doDeTokenizeAsString(long pointer, IntBuffer buf, int pos, int size,
			boolean removeSpecial, boolean unparseSpecial);

	/*
	 * API
	 */

	public void tokenize(CharSequence str, IntBuffer tokens, boolean addSpecial, boolean parseSpecial) {
		if (stringMode) {
			tokenizeUtf16(str, tokens, addSpecial, parseSpecial);
		} else {
			CharBuffer chars = CharBuffer.wrap(str);
			ByteBuffer utf8 = UTF_8.encode(chars);
			tokenizeUtf8(utf8, tokens, addSpecial, parseSpecial);
		}
	}

	public IntBuffer tokenize(CharSequence str, boolean addSpecial, boolean parseSpecial) {
		int[] arr;
		if (stringMode) {
			arr = tokenizeUtf16(str.toString(), addSpecial, parseSpecial);
		} else {
			CharBuffer chars = CharBuffer.wrap(str);
			ByteBuffer utf8 = UTF_8.encode(chars);
			arr = tokenizeUtf8(utf8, addSpecial, parseSpecial);
		}
		// return IntBuffer.wrap(arr).asReadOnlyBuffer();
		return IntBuffer.wrap(arr);
	}

	public void tokenize(ByteBuffer utf8, IntBuffer tokens, boolean addSpecial, boolean parseSpecial)
			throws IndexOutOfBoundsException {
		if (stringMode) {
			CharBuffer chars = UTF_8.decode(utf8);
			tokenizeUtf16(chars.toString(), tokens, addSpecial, parseSpecial);
		} else {
			tokenizeUtf8(utf8, tokens, addSpecial, parseSpecial);
		}
	}

	public IntBuffer tokenize(ByteBuffer utf8, boolean addSpecial, boolean parseSpecial) {
		int[] arr;
		if (stringMode) {
			CharBuffer chars = UTF_8.decode(utf8);
			arr = tokenizeUtf16(chars.toString(), addSpecial, parseSpecial);
		} else {
			arr = tokenizeUtf8(utf8, addSpecial, parseSpecial);
		}
		return IntBuffer.wrap(arr);
	}

	public void deTokenize(IntBuffer in, ByteBuffer utf8, boolean removeSpecial, boolean unparseSpecial)
			throws IndexOutOfBoundsException {
		if (stringMode) {
			String s = deTokenizeUtf16(in, removeSpecial, unparseSpecial);
			byte[] bytes = s.getBytes(UTF_8);
			if (bytes.length > utf8.remaining())
				throw new IndexOutOfBoundsException(bytes.length);
			utf8.put(bytes);
		} else {
			deTokenizeUtf8(in, utf8, removeSpecial, unparseSpecial);
		}
	}

	public String deTokenize(IntBuffer in, boolean removeSpecial, boolean unparseSpecial) {
		if (stringMode) {
			return deTokenizeUtf16(in, removeSpecial, unparseSpecial);
		} else {
			byte[] bytes = deTokenizeUtf8(in, removeSpecial, unparseSpecial);
			return new String(bytes, UTF_8);
		}
	}

	/*
	 * DEFAULTS
	 */
	final public IntBuffer[] tokenizeMultiple(List<? extends CharSequence> prompts) {
		IntBuffer[] tokenLists = new IntBuffer[prompts.size()];
		for (int i = 0; i < prompts.size(); i++) {
			CharSequence prompt = prompts.get(i);
			IntBuffer tokenList = tokenize(prompt);
			tokenLists[i] = tokenList;
		}
		return tokenLists;
	}

	final public IntBuffer tokenize(CharSequence str) {
		return tokenize(str, false, true);
	}

	final public void tokenize(CharSequence str, IntBuffer tokens) throws IndexOutOfBoundsException {
		tokenize(str, tokens, false, true);
	}

	final public String deTokenize(IntBuffer in) {
		return deTokenize(in, true, true);
	}

	final public void deTokenize(IntBuffer in, ByteBuffer out) throws IndexOutOfBoundsException {
		deTokenize(in, out, true, true);
	}

	/*
	 * UTF-8
	 */

	int[] tokenizeUtf8(ByteBuffer in, boolean addSpecial, boolean parseSpecial) {
		checkInput(in);
		// ensure position is 0
		// ByteBuffer in = str.slice().limit(str.limit() - str.position());
		synchronized (in) {
			int[] tokenArr;
			if (in.isDirect()) {
				tokenArr = doTokenizeUtf8AsArray(model.getAsLong(), in, in.position(), in.remaining(), addSpecial,
						parseSpecial);
				in.position(in.limit());
			} else if (in.hasArray() && !in.isReadOnly()) {
				byte[] arr = in.array();
				tokenArr = doTokenizeUtf8BytesAsArray(model.getAsLong(), arr, in.arrayOffset(), in.remaining(),
						addSpecial, parseSpecial);
				in.position(in.limit());
			} else {// copy
				byte[] copy = new byte[in.remaining()];
				in.get(copy, in.position(), copy.length);
				tokenArr = doTokenizeUtf8BytesAsArray(model.getAsLong(), copy, 0, copy.length, addSpecial,
						parseSpecial);
			}
			return tokenArr;
		}
	}

	void tokenizeUtf8(ByteBuffer str, IntBuffer tokens, boolean addSpecial, boolean parseSpecial)
			throws IndexOutOfBoundsException {
		checkInput(str);
		checkOutput(tokens);
		synchronized (tokens) {// we are writing into this buffer and changing its position
			if (str.isDirect() && tokens.isDirect()) {// optimal
				int count = doTokenizeUtf8(model.getAsLong(), str, str.position(), str.remaining(), tokens,
						tokens.position(), tokens.remaining(), addSpecial, parseSpecial);
				if (count < 0)
					throw new IndexOutOfBoundsException(-count);
				str.position(str.limit());
				tokens.position(tokens.position() + count);
			} else {
				int[] tokenArr = tokenizeUtf8(str, addSpecial, parseSpecial);
				if (tokenArr.length > tokens.remaining())
					throw new IndexOutOfBoundsException(tokenArr.length);
				tokens.put(tokenArr);
			}
		}
	}

	byte[] deTokenizeUtf8(IntBuffer in, boolean removeSpecial, boolean unparseSpecial) {
		byte[] outArr;
		if (in.isDirect()) {
			outArr = doDeTokenizeAsUtf8Bytes(model.getAsLong(), in, in.position(), in.remaining(), removeSpecial,
					unparseSpecial);
			in.position(in.limit());
		} else if (in.hasArray() && !in.isReadOnly()) {
			outArr = doDeTokenizeArrayAsUtf8Bytes(model.getAsLong(), in.array(), in.arrayOffset(), in.remaining(),
					removeSpecial, unparseSpecial);
			in.position(in.limit());
		} else {// copy
			int[] copy = new int[in.remaining()];
			in.get(copy, in.position(), copy.length);
			outArr = doDeTokenizeArrayAsUtf8Bytes(model.getAsLong(), copy, 0, copy.length, removeSpecial,
					unparseSpecial);
		}
		return outArr;
	}

	void deTokenizeUtf8(IntBuffer in, ByteBuffer str, boolean removeSpecial, boolean unparseSpecial)
			throws IndexOutOfBoundsException {
		if (in.isDirect() && str.isDirect()) {
			int count = doDeTokenizeAsUtf8(model.getAsLong(), in, in.position(), in.remaining(), str, str.position(),
					str.remaining(), removeSpecial, unparseSpecial);
			if (count < 0)
				throw new IndexOutOfBoundsException(-count);
			str.position(str.position() + count);
			in.position(in.limit());
		} else {
			byte[] bytes = deTokenizeUtf8(in, removeSpecial, unparseSpecial);
			if (bytes.length > (str.limit() - str.position()))
				throw new IndexOutOfBoundsException(bytes.length);
			str.put(bytes);
		}
	}

	/*
	 * UTF-16
	 */
	int[] tokenizeUtf16(String str, boolean addSpecial, boolean parseSpecial) {
		return doTokenizeStringAsArray(model.getAsLong(), str, addSpecial, parseSpecial);
	}

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
	void tokenizeUtf16(CharSequence str, IntBuffer tokens, boolean addSpecial, boolean parseSpecial) {
		Objects.requireNonNull(str);
		checkOutput(tokens);
		synchronized (tokens) {// we are writing into this buffer and changing its position
//			IntBuffer tokensToUse = tokens.slice().limit(tokens.limit() - tokens.position());
			String in = str.toString();
			assert str instanceof String ? in == str : true;
			int[] tokenArr = doTokenizeStringAsArray(model.getAsLong(), in, addSpecial, parseSpecial);
			if (tokenArr.length > (tokens.limit() - tokens.position()))
				throw new IndexOutOfBoundsException(tokenArr.length);
			tokens.put(tokenArr);
		}
	}

	String deTokenizeUtf16(IntBuffer in, boolean removeSpecial, boolean unparseSpecial) {
		Objects.requireNonNull(in);
		if (in.isDirect()) {
			String res = doDeTokenizeAsString(model.getAsLong(), in, in.position(), in.remaining(), removeSpecial,
					unparseSpecial);
			in.position(in.limit());
			return res;
		} else {
			String res;
			if (in.hasArray()) {
				res = doDeTokenizeArrayAsString(model.getAsLong(), in.array(), in.arrayOffset(), in.remaining(),
						removeSpecial, unparseSpecial);
				in.position(in.limit());
			} else {// copy
				int[] copy = new int[in.limit() - in.position()];
				in.get(copy, in.position(), copy.length);
				res = doDeTokenizeArrayAsString(model.getAsLong(), copy, 0, copy.length, removeSpecial, unparseSpecial);
			}
			return res;
		}
	}

	/*
	 * UTILITIES
	 */
	private void checkInput(Buffer in) {
		if (in instanceof IntBuffer)
			if (!ByteOrder.nativeOrder().equals(((IntBuffer) in).order()))
				throw new IllegalArgumentException("Int buffer does not use native byte order");
		Objects.requireNonNull(in, "Input buffer cannot be null");
	}

	private void checkOutput(Buffer out) {
		Objects.requireNonNull(out, "Output buffer cannot be null");
		if (out.isReadOnly())
			throw new IllegalArgumentException("Output buffer is read-only");
		if (out instanceof IntBuffer)
			if (!ByteOrder.nativeOrder().equals(((IntBuffer) out).order()))
				throw new IllegalArgumentException("Int buffer does not use native byte order");
	}
	/*
	 * ACCESSORS
	 */

	synchronized boolean isStringMode() {
		return stringMode;
	}

	synchronized void setStringMode(boolean stringMode) {
		this.stringMode = stringMode;
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
