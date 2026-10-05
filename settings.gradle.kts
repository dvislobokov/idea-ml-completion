rootProject.name = "idea-ml-completion"

pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories { mavenCentral() }
}

include("ml-core", "ml-train")
