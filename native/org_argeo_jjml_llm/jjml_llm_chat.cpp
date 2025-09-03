#include <string>
#include <vector>
#include <cstring>
#include <cassert>

#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_llm_LLamaCppNativeChatFormatter.h" // IWYU pragma: keep

/*
 * CHAT
 */
JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_llm_LLamaCppNativeChatFormatter_doFormatChatMessages(
		JNIEnv *env, jclass, jobjectArray roles, jobjectArray contents,
		jboolean addAssistantTokens, jbyteArray chatTemplateStr) {
	const jsize messages_size = env->GetArrayLength(roles);
	assert(env->GetArrayLength(contents) == messages_size);

	std::vector<llama_chat_message> chat_messages;

	try {
		int alloc_size = 0;
		// since the content can be quite big, we go through the heap
		for (int i = 0; i < messages_size; i++) {
			std::string u8_role = argeo::jni::to_string(env, roles, i);
			std::string u8_content = argeo::jni::to_string(env, contents, i);

			char *role = new char[u8_role.length() + 1];
			strcpy(role, u8_role.c_str());

			char *content = new char[u8_content.length() + 1];
			strcpy(content, u8_content.c_str());

			llama_chat_message message { role, content };
			chat_messages.push_back(message);

			// using the same factor as in common.cpp
			alloc_size += (u8_role.length() + u8_content.length()) * 1.25;
		}

		std::string u8_chat_template;
		if (chatTemplateStr != nullptr)
			u8_chat_template = argeo::jni::to_string(env, chatTemplateStr);

		std::vector<char> buf(alloc_size);
		int32_t resLength = llama_chat_apply_template(
				chatTemplateStr != nullptr ? u8_chat_template.c_str() : nullptr,
				chat_messages.data(), chat_messages.size(), addAssistantTokens,
				buf.data(), buf.size());

		// error: chat template is not supported
		if (resLength < 0) {
			if (chatTemplateStr != nullptr)
				throw std::runtime_error("Custom template is not supported");
			else
				throw std::runtime_error("Built-in template is not supported");
		}

		// if it turns out that our buffer is too small, we resize it
		if ((size_t) resLength > buf.size()) {
			buf.resize(resLength);
			resLength = llama_chat_apply_template(
					chatTemplateStr != nullptr ?
							u8_chat_template.c_str() : nullptr,
					chat_messages.data(), chat_messages.size(),
					addAssistantTokens, buf.data(), buf.size());
		}

		// we clean up, since we don't need the messages anymore
		for (int i = 0; i < messages_size; i++) {
			llama_chat_message message = chat_messages[i];
			delete message.role;
			delete message.content;
		}

		std::string u8_res(buf.data(), resLength);
		jbyteArray res = env->NewByteArray(u8_res.length());
		env->SetByteArrayRegion(res, 0, u8_res.length(), (jbyte*) &u8_res[0]);
		return res;
	} catch (std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}
