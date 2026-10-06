// Pure Kotlin, no IntelliJ dependencies: tokens, vocabulary, n-gram LM, ranking model, feature extraction, model and shard
// formats, metrics. Embedded into IDE plugins via git subtree (docs/ADAPTER.md), so this build file is self-sufficient:
// it does not rely on the root project's configuration and must not pull third-party runtime libraries
// (class-loader conflicts, plugin size). The Kotlin plugin version comes from the host's root `plugins` block.
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinVersion

plugins {
    id("org.jetbrains.kotlin.jvm")
}

repositories {
    mavenCentral()
}

kotlin {
    compilerOptions {
        // Same as the IntelliJ plugins: the platform bundles its own Kotlin stdlib (2.3.x), don't use newer API.
        apiVersion.set(KotlinVersion.KOTLIN_2_3)
        languageVersion.set(KotlinVersion.KOTLIN_2_3)
        jvmTarget.set(JvmTarget.JVM_21)
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

// Optional jdk.incubator.vector kernels (io.github.completionml.core.nn.vector): a separate source set compiled with
// the incubator module, packed into the same jar and only loaded reflectively when the VM was started with
// `--add-modules jdk.incubator.vector` (NnKernels.best()). Main code never references it, so ml-core works without it.
val vector: SourceSet by sourceSets.creating {
    compileClasspath += sourceSets.main.get().output + sourceSets.main.get().compileClasspath
}
kotlin.target.compilations.named("vector") {
    compileTaskProvider.configure { compilerOptions.freeCompilerArgs.add("-Xadd-modules=jdk.incubator.vector") }
}
tasks.named<Jar>("jar") { from(vector.output) }
sourceSets.test {
    compileClasspath += vector.output
    runtimeClasspath += vector.output
}

dependencies {
    // `kotlin.stdlib.default.dependency = false` in the plugins: declare it explicitly, the host provides the real one at runtime
    implementation(kotlin("stdlib"))
    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    maxHeapSize = "2g"
    testLogging { events("failed"); showStackTraces = true; exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL }
}

// The same tests with the Vector API module resolved (the default `test` task runs without it → plain kernels).
val vectorTest by tasks.registering(Test::class) {
    description = "Runs the nn tests with --add-modules jdk.incubator.vector"
    group = "verification"
    testClassesDirs = sourceSets.test.get().output.classesDirs
    classpath = sourceSets.test.get().runtimeClasspath
    jvmArgs("--add-modules=jdk.incubator.vector")
    systemProperty("completionml.nn.expectVector", "true")
    filter { includeTestsMatching("io.github.completionml.core.nn.*") }
}

// Prints the runtime classpath for running the benchmark main (io.github.completionml.core.nn.NnBenchKt) with plain `java`.
val printBenchClasspath by tasks.registering {
    val cp = sourceSets.test.get().runtimeClasspath
    dependsOn(tasks.named("testClasses"))
    doLast { println("CP=" + cp.asPath) }
}
