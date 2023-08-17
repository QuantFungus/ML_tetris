using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class DataManager : MonoBehaviour
{
    // Start is called before the first frame update

    //used for determining whether to spawn a maze or a labyrinth
    public static bool mOL;

    //used to determine whether player wants WASD movement or mouse controlled movement
    public static bool mousePlayer = false;

    //used in game selection to determine the size of the grid
    public static int rows = 0;
    public static int cols = 0;

    public GameObject toggle;

    void Start()
    {
        mousePlayer = false;
       // toggle = GameObject.FindGameObjectWithTag("toggle");
    }

    // Update is called once per frame
    void Update()
    {
        //mousePlayer = toggle.GetComponent<Toggle>().isOn;
    }

    //used in inputfield to get row data
    public void setRows(string n)
    {
        print(n);
        int.TryParse(n, out rows);
    }

    //used in inputfield to get col data
    public void setCols(string n)
    {
        print(n);
        int.TryParse(n, out cols);
    }

    public void changeMousePlayer()
    {
        if(mousePlayer == false)
        {
            mousePlayer = true;
        }
        else
        {
            mousePlayer = false;
        }
    }
}
