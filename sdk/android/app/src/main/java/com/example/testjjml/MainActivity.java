package com.example.testjjml;

import android.os.Bundle;
import android.view.MenuItem;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;

import com.example.testjjml.databinding.ActivityMainBinding;
import com.google.android.material.navigation.NavigationBarView;

import org.argeo.jjml.llm.LlamaCppNative;
import org.argeo.jjml.mtmd.MtmdNative;
import org.argeo.jjml.whisper.WhisperNative;

public class MainActivity extends AppCompatActivity {

    static {
        LlamaCppNative.ensureLibrariesLoaded();
        System.out.println("MTMD: " + MtmdNative.isAvailable());
        System.out.println("Whisper: " + WhisperNative.isAvailable());
    }

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Start loading the LLM in the background
        LlmManager.INSTANCE.load(this);

        binding.bottomNavigation.setOnItemSelectedListener(new NavigationBarView.OnItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(@NonNull MenuItem item) {
                int itemId = item.getItemId();
                if (itemId == R.id.nav_chat) {
                    loadFragment(new ChatFragment());
                    return true;
                } else if (itemId == R.id.nav_image) {
                    loadFragment(new ImageFragment());
                    return true;
                } else if (itemId == R.id.nav_audio) {
                    loadFragment(new AudioFragment());
                    return true;
                }
                return false;
            }
        });

        // Load the chat fragment by default
        if (savedInstanceState == null) {
            binding.bottomNavigation.setSelectedItemId(R.id.nav_chat);
        }
    }

    private void loadFragment(Fragment fragment) {
        getSupportFragmentManager()
                .beginTransaction()
                .replace(R.id.fragment_container, fragment)
                .commit();
    }
}
