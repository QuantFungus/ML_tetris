using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class Buttonz : MonoBehaviour
{
    //public static bool mOrl;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    //used for game win screen - same as begin() for now but planning to add additional features to this button function later
    public void testRetry()
    {
        SceneManager.LoadScene("GameMenu");
    }

    //used for startup menu button
    public void begin()
    {
        SceneManager.LoadScene("GameMenu");
    }

    //used for the red X exit button
    public void exit()
    {
        Application.Quit();
    }

    //used for the game selection buttons
    public void MazeGame(bool mazeOrLaby)
    {
        DataManager.mOL = mazeOrLaby;
        SceneManager.LoadScene("MazeCanvasTest");
    }
}
