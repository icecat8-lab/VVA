# The native library looks these up by name through JNI; keep them when the app is minified.
-keep class com.vva.engine.VVAEngine { *; }
-keep interface com.vva.engine.VVAEngine$ProgressListener { *; }
