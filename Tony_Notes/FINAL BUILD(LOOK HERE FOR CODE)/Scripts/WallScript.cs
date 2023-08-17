using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WallScript : MonoBehaviour
{
    //visited bool for DFS
    private bool visited = false;

    //used for storing the wall's index in the grid data structure
    public int colIndex = 0;
    public int rowIndex = 0;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    //sets visit to true
    public void Visit()
    {
        visited = true;
    }

    //checks the value of visit
    public bool isVisited()
    {
        if (visited)
        {
            return true;
        }
        return false;
    }

    //sets the index of the wall node
    public void setIndex(int i, int j)
    {
        colIndex = i;
        rowIndex = j;
    }

    //returns the column index
    public int getCol()
    {
        return colIndex;
    }

    //returns the row index
    public int getRow()
    {
        return rowIndex;
    }
}
