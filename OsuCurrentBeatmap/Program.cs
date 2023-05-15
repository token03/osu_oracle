using System;
using System.Diagnostics;
using System.Net.Http;
using System.Net.Http.Json;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using OsuMemoryDataProvider;

namespace OsuCurrentBeatmap
{
    internal class Program
    {
        static HttpClient client = new HttpClient();
        static CancellationTokenSource cts = new CancellationTokenSource();

        static void Main(string[] args)
        {
            string osuWindowTitleHint = ""; // You can specify a hint here if needed
            string assemblyLocation = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            string scriptPath = Path.Combine("python", "predict.py");
            string modelPath = Path.Combine("python", "models");

            // Get the username of the current user
            string username = Environment.UserName;

            // Get the path to the osu! config file
            string configFilePath = Path.Combine("C:\\Users", username, "AppData\\Local\\osu!\\osu!." + username + ".cfg");

            // Read the config file and find the beatmap directory
            string beatmapDirectory = null;
            using (StreamReader sr = new StreamReader(configFilePath))
            {
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    if (line.StartsWith("BeatmapDirectory = "))
                    {
                        beatmapDirectory = line.Substring("BeatmapDirectory = ".Length);
                        break;
                    }
                }
            }

            if (beatmapDirectory == null)
            {
                throw new Exception("Could not find beatmap directory in osu! config file.");
            }
#pragma warning disable CS0618 // Type or member is obsolete
            IOsuMemoryReader reader = OsuMemoryReader.Instance.GetInstanceForWindowTitleHint(osuWindowTitleHint);
#pragma warning restore CS0618 // Type or member is obsolete

            // Start the Python service
            Process pythonService = new Process();
            pythonService.StartInfo.FileName = "py";  // Or the path to your Python interpreter
            pythonService.StartInfo.Arguments = scriptPath;
            pythonService.Start();

            CancellationTokenSource cts = new CancellationTokenSource();
            string previousMapFile = null;
            Task<string> predictionTask = null;

            while (true)
            {
                if (cts.Token.IsCancellationRequested)
                    break;

                var status = reader.GetCurrentStatus(out _);
                if (status == OsuMemoryStatus.SongSelect || status == OsuMemoryStatus.SongSelectEdit)
                {
                    string mapFolderName = reader.GetMapFolderName();
                    string osuFileName = reader.GetOsuFileName();

                    string mapFilePath = Path.Combine(beatmapDirectory, mapFolderName, osuFileName);

                    if (mapFilePath != previousMapFile)
                    {
                        Console.Clear();
                        Console.WriteLine($"Previewing beatmap file: {mapFilePath}");

                        // Cancel the previous prediction task
                        if (predictionTask != null)
                        {
                            cts.Cancel();
                            cts = new CancellationTokenSource();  // Create a new cancellation token source
                        }

                        previousMapFile = mapFilePath;

                        // Start a new prediction task
                        predictionTask = Predict(new string[] { modelPath }, mapFilePath, cts.Token);
                        predictionTask.ContinueWith(task =>
                        {
                            if (task.IsCompletedSuccessfully)
                            {
                                Console.WriteLine($"Prediction: {task.Result}");
                            }
                        });
                    }
                }

                Thread.Sleep(1000); // Add delay to prevent constant polling
            }

            // Stop the Python service
            pythonService.Kill();
        }

        static async Task<string> Predict(string[] folders, string mapFilePath, CancellationToken cancellationToken)
        {
            var requestBody = new
            {
                map_file_path = mapFilePath,
                folders = folders
            };

            var response = await client.PostAsJsonAsync("http://localhost:5000/predict", requestBody, cancellationToken);

            if (response.IsSuccessStatusCode)
            {
                var responseBody = await response.Content.ReadAsStringAsync();
                return responseBody;
            }
            else
            {
                throw new Exception($"Prediction request failed: {response.StatusCode}");
            }
        }
    }
}
