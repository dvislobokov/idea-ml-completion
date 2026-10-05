// Pure Kotlin, no IntelliJ dependencies: tokens, vocabulary, n-gram LM, ranking model, model format, metrics.
// Embedded into IDE plugins; must not pull third-party runtime libraries (class-loader conflicts, plugin size).
dependencies {
    implementation(kotlin("stdlib"))
}
