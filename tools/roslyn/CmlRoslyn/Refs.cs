using System.Text.Json;
using Microsoft.CodeAnalysis;

namespace CmlRoslyn;

/// <summary>Metadata references for repositories without project files: the installed reference packs
/// (Microsoft.NETCore.App.Ref, Microsoft.AspNetCore.App.Ref), the Windows desktop ref pack and every compile-time
/// assembly of the synthetic <c>refpacks/RefPacks.csproj</c> restore (read from its <c>project.assets.json</c>).
/// Assemblies are de-duplicated by file name, first occurrence wins (ref packs first).</summary>
public static class Refs
{
    public static List<MetadataReference> Load(string? dotnetRoot, string? assetsJson, string? extraDirs, TextWriter log)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var files = new List<string>();
        void Add(string path)
        {
            var name = Path.GetFileName(path);
            if (!name.EndsWith(".dll", StringComparison.OrdinalIgnoreCase)) return;
            if (name.Contains("Microsoft.VisualBasic", StringComparison.OrdinalIgnoreCase)) return;
            if (seen.Add(name)) files.Add(path);
        }
        void AddDir(string dir)
        {
            if (!Directory.Exists(dir)) { log.WriteLine($"refs: missing {dir}"); return; }
            foreach (var f in Directory.GetFiles(dir, "*.dll").OrderBy(x => x, StringComparer.Ordinal)) Add(f);
        }

        dotnetRoot ??= FindDotnetRoot();
        if (dotnetRoot != null)
        {
            foreach (var pack in new[] { "Microsoft.NETCore.App.Ref", "Microsoft.AspNetCore.App.Ref" })
            {
                var packDir = Path.Combine(dotnetRoot, "packs", pack);
                if (!Directory.Exists(packDir)) { log.WriteLine($"refs: no pack {packDir}"); continue; }
                var ver = Directory.GetDirectories(packDir).OrderByDescending(x => x, StringComparer.Ordinal).First();
                var refDir = Directory.GetDirectories(Path.Combine(ver, "ref")).OrderByDescending(x => x, StringComparer.Ordinal).First();
                AddDir(refDir);
            }
        }
        var nuget = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nuget", "packages");
        var desktop = Path.Combine(nuget, "microsoft.windowsdesktop.app.ref");
        if (Directory.Exists(desktop))
        {
            var ver = Directory.GetDirectories(desktop).OrderByDescending(x => x, StringComparer.Ordinal).First();
            var refRoot = Path.Combine(ver, "ref");
            if (Directory.Exists(refRoot)) AddDir(Directory.GetDirectories(refRoot).OrderByDescending(x => x, StringComparer.Ordinal).First());
        }
        if (assetsJson != null && File.Exists(assetsJson))
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(assetsJson));
            var root = doc.RootElement;
            var folders = root.GetProperty("packageFolders").EnumerateObject().Select(p => p.Name).ToList();
            var targets = root.GetProperty("targets");
            var target = targets.EnumerateObject().First().Value;
            foreach (var pkg in target.EnumerateObject())
            {
                if (!pkg.Value.TryGetProperty("compile", out var compile)) continue;
                var idVer = pkg.Name.Split('/');
                foreach (var asm in compile.EnumerateObject())
                {
                    if (!asm.Name.EndsWith(".dll", StringComparison.OrdinalIgnoreCase)) continue;
                    foreach (var folder in folders)
                    {
                        var path = Path.Combine(folder, idVer[0].ToLowerInvariant(), idVer[1].ToLowerInvariant(), asm.Name.Replace('/', Path.DirectorySeparatorChar));
                        if (File.Exists(path)) { Add(path); break; }
                    }
                }
            }
        }
        if (extraDirs != null)
            foreach (var d in extraDirs.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)) AddDir(d);

        var refs = new List<MetadataReference>(files.Count);
        foreach (var f in files)
        {
            try { refs.Add(MetadataReference.CreateFromFile(f)); }
            catch (Exception e) { log.WriteLine($"refs: skip {f}: {e.Message}"); }
        }
        log.WriteLine($"refs: {refs.Count} assemblies");
        return refs;
    }

    static string? FindDotnetRoot()
    {
        var env = Environment.GetEnvironmentVariable("DOTNET_ROOT");
        if (env != null && Directory.Exists(env)) return env;
        foreach (var c in new[] { "/usr/lib/dotnet", "/usr/share/dotnet", "/usr/local/share/dotnet" })
            if (Directory.Exists(Path.Combine(c, "packs"))) return c;
        var dir = Path.GetDirectoryName(typeof(object).Assembly.Location);
        // …/dotnet/shared/Microsoft.NETCore.App/x.y.z
        for (var i = 0; i < 3 && dir != null; i++) dir = Path.GetDirectoryName(dir);
        return dir != null && Directory.Exists(Path.Combine(dir, "packs")) ? dir : null;
    }
}
