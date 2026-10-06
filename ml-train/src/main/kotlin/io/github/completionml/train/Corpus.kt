package io.github.completionml.train

import io.github.completionml.core.spi.MlLanguage
import java.io.File

/**
 * Source files of a language under `<data>/repos/<owner>__<repo>/...`, with generated/vendored files dropped and a
 * deterministic train/test split (by repository when there are enough of them, otherwise by file hash).
 */
class Corpus(val root: File, val language: MlLanguage, val splitBy: String = "auto", val testShare: Int = 10, includeList: File? = null,
             /** explicit held-out repositories (directory names); overrides the hash split */
             val testRepos: Set<String> = emptySet()) {
    class Source(val repo: String, val file: File)

    /** Repositories under `repos/`, optionally restricted to the directory names listed in [includeList] (one per line). */
    val repos: List<File> = run {
        val all = File(root, "repos").listFiles { f -> f.isDirectory }?.sortedBy { it.name } ?: emptyList()
        if (includeList == null) all else {
            val names = includeList.readLines().map { it.trim() }.filter { it.isNotEmpty() && !it.startsWith("#") }.toSet()
            val missing = names - all.map { it.name }.toSet()
            require(missing.isEmpty()) { "repos listed in $includeList but not cloned: $missing" }
            all.filter { it.name in names }
        }
    }

    fun files(): List<Source> = repos.flatMap { repo ->
        repo.walkTopDown()
            .onEnter { d -> d.name !in EXCLUDED_DIRS && d.name != ".git" && !isSymlink(d) }   // cloned repos are untrusted: never follow links out of the tree
            .filter { f -> f.isFile && !isSymlink(f) && language.extensions.any { f.name.endsWith(it) } && f.length() in 1..MAX_FILE_BYTES && !isGenerated(f) }
            .map { Source(repo.name, it) }
            .toList()
    }

    fun isTest(s: Source): Boolean {
        if (testRepos.isNotEmpty()) return s.repo in testRepos
        val by = if (splitBy == "auto") (if (repos.size >= 20) "repo" else "file") else splitBy
        val key = if (by == "repo") s.repo else s.repo + "/" + s.file.relativeTo(root).path.replace('\\', '/')
        val h = key.hashCode().toLong() and 0x7fffffffL
        return (h % 100) < testShare
    }

    private fun isSymlink(f: File) = java.nio.file.Files.isSymbolicLink(f.toPath())

    private fun isGenerated(f: File): Boolean {
        if (isGeneratedName(f.name)) return true
        // first 2 KB: auto-generated markers
        val head = ByteArray(2048)
        val read = f.inputStream().use { it.read(head) }
        if (read <= 0) return true
        return isGeneratedHead(String(head, 0, read, Charsets.UTF_8))
    }

    companion object {
        const val MAX_FILE_BYTES = 1L shl 20
        val EXCLUDED_DIRS = setOf("bin", "obj", "vendor", "third_party", "external", "node_modules", "packages", "testdata",
            "PackageCache",      // Unity Library/PackageCache: vendored UPM packages
            "Generated")         // google-api-dotnet-client, Silk.NET, protobuf/grpc output dirs

        fun isExcludedDir(name: String) = name in EXCLUDED_DIRS || name == ".git"

        fun isGeneratedName(n: String): Boolean {
            if (n.endsWith(".pb.go") || n.endsWith("_string.go")) return true
            val l = n.lowercase()
            if (l.endsWith(".g.cs") || l.endsWith(".g.i.cs") || l.endsWith(".generated.cs") || l.endsWith(".gen.cs")) return true   // Silk.NET `Sdl.gen.cs` has no header marker
            if (l.endsWith(".designer.cs")) return true                                           // WinForms/Settings/Resources designers (lower-case: Xamarin, MonoDevelop)
            if (l.endsWith("modelsnapshot.cs") || l.endsWith("assemblyattributes.cs")) return true  // EF Core snapshot; MSBuild assembly attributes
            if (l.endsWith(".verified.cs") || l.endsWith(".received.cs") || l.endsWith(".approved.cs")) return true   // Verify / ApprovalTests snapshots
            if (n.startsWith("TemporaryGeneratedFile_")) return true
            return l.endsWith("assemblyinfo.cs") || l.endsWith("globalusings.cs")
        }

        /** EF Core migrations `Migrations/20240101123456_Name.cs` are scaffolded. [rel] is the path inside the repository with `/`. */
        private val EF_MIGRATION = Regex("""(^|/)Migrations/\d{8,}_[^/]*\.cs$""")

        fun isGeneratedPath(rel: String): Boolean = EF_MIGRATION.containsMatchIn(rel)

        /** [head] is the first ~2 KB of a file as text. */
        fun isGeneratedHead(head: String): Boolean {
            // markers count only inside comments: generator sources print them from string literals (golang/go cmd/cgo/godefs.go)
            var auto = false; var generated = false; var doNotEdit = false; var generatedBy = false
            for (raw in head.removePrefix("﻿").lineSequence()) {    // trimStart() does not strip a BOM
                val line = raw.trimStart()
                if (!(line.startsWith("//") || line.startsWith("/*") || line.startsWith("*") || line.startsWith("#"))) continue
                if (line.contains("<auto-generated")) auto = true
                if (line.contains("Code generated") || line.contains("Generated code")) generated = true   // google-api: "// Generated code. DO NOT EDIT!"
                // phrases that are unambiguous on their own (no DO NOT EDIT line): AutoRest, AWS SDK, T4 / Roslyn generators,
                // openapi/swagger clients, decompiler output
                if (line.contains("Code generated by") || line.contains("Do not modify this file. This file is generated") ||
                    line.contains("lost if the code is regenerated") || line.contains("Changes to this file may cause incorrect behavior") ||
                    line.contains("openapi-generator") || line.contains("swagger-codegen") || line.contains("Generated by: https://") ||
                    line.contains("Decompiled with") || line.contains("decompiled by") || line.contains("ILSpy") || line.contains("dnSpy") ||
                    line.contains("Il2CppDumper") || line.contains("Il2CppInspector")) generatedBy = true
                if (line.contains("DO NOT EDIT") || line.contains("do not edit") || line.contains("Do not edit") || line.contains("do not modify")) doNotEdit = true
            }
            return auto || generated && doNotEdit || generatedBy
        }
    }
}
