using System;
using System.Diagnostics;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading;
using OsuMemoryDataProvider;

namespace OsuCurrentBeatmap
{
    internal class Program
    {
        static HttpClient client = new HttpClient();
        static void Main(string[] args)
        {
            string osuWindowTitleHint = ""; // You can specify a hint here if needed

            // Get instance of memory reader
#pragma warning disable CS0618 // Type or member is obsolete
            IOsuMemoryReader reader = OsuMemoryReader.Instance.GetInstanceForWindowTitleHint(osuWindowTitleHint);
#pragma warning restore CS0618 // Type or member is obsolete

            // Start the Python service
            Process pythonService = new Process();
            pythonService.StartInfo.FileName = "py";  // Or the path to your Python interpreter
            pythonService.StartInfo.Arguments = "C:\\Users\\Jessie\\Desktop\\osu_oracle\\OsuCurrentBeatmap\\python\\predict.py";
            pythonService.Start();

            CancellationTokenSource cts = new CancellationTokenSource();
            int previousMapId = -1;

            // This is a simple loop to continuously poll for the current beatmap ID.
            // You might want to add a delay or other stopping condition.
            while (true)
            {
                if (cts.IsCancellationRequested)
                    break;

                var status = reader.GetCurrentStatus(out _);
                if (status == OsuMemoryStatus.SongSelect)
                {
                    var mapId = reader.GetMapId();
                    if (mapId != previousMapId)
                    {
                        Console.Clear();
                        Console.WriteLine($"Previewing beatmap id: {mapId}");
                        previousMapId = mapId;

                        // Make a prediction
                        var prediction = Predict(new string[] { "C:\\Users\\Jessie\\Desktop\\osu_oracle\\OsuCurrentBeatmap\\python\\models" }, mapId).GetAwaiter().GetResult();  // Replace with your actual folders
                        Console.WriteLine($"Prediction: {prediction}");
                    }
                }

                Thread.Sleep(1000); // Add delay to prevent constant polling
            }

            // Stop the Python service
            pythonService.Kill();
        }

        static async Task<string> Predict(string[] folders, int beatmapId)
        {
            var requestBody = new
            {
                beatmap_id = beatmapId,
                folders = folders
            };

            var response = await client.PostAsJsonAsync("http://localhost:5000/predict", requestBody);

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
