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


            // Get instance of memory reader
#pragma warning disable CS0618 // Type or member is obsolete
            IOsuMemoryReader reader = OsuMemoryReader.Instance.GetInstanceForWindowTitleHint(osuWindowTitleHint);
#pragma warning restore CS0618 // Type or member is obsolete

            // Start the Python service
            Process pythonService = new Process();
            pythonService.StartInfo.FileName = "py";  // Or the path to your Python interpreter
            pythonService.StartInfo.Arguments = scriptPath;
            pythonService.Start();

            int previousMapId = -1;
            Task<string> predictionTask = null;

            // This is a simple loop to continuously poll for the current beatmap ID.
            // You might want to add a delay or other stopping condition.
            while (true)
            {
                if (cts.Token.IsCancellationRequested)
                    break;

                var status = reader.GetCurrentStatus(out _);
                if (status == OsuMemoryStatus.SongSelect || status == OsuMemoryStatus.SongSelectEdit)
                {
                    var mapId = reader.GetMapId();
                    if (mapId != previousMapId)
                    {
                        Console.Clear();
                        Console.WriteLine($"Previewing beatmap id: {mapId}");

                        // Cancel the previous prediction task
                        if (predictionTask != null)
                        {
                            cts.Cancel();
                            cts = new CancellationTokenSource();  // Create a new cancellation token source
                        }

                        previousMapId = mapId;

                        // Start a new prediction task
                        predictionTask = Predict(new string[] { modelPath }, mapId, cts.Token);
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

        static async Task<string> Predict(string[] folders, int beatmapId, CancellationToken cancellationToken)
        {
            var requestBody = new
            {
                beatmap_id = beatmapId,
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
