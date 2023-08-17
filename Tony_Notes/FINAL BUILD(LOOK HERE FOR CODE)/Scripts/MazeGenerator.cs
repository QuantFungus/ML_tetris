using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class MazeGenerator : MonoBehaviour
{
    public GameObject wallBlock;
    public PlayerMove playerMoveScript;
    private GameObject[,] theWalls;
    private int rows;
    private int cols;
    private int labybool;
    public bool Laby;
    public bool useMouse;
    Vector3 starterPos;
    // Start is called before the first frame update
    void Start()
    {

        Laby = DataManager.mOL;
        rows = DataManager.rows;
        cols = DataManager.cols;
        useMouse = DataManager.mousePlayer;
        if(rows < 2)
        {
            rows = 2;
        }
        else if(rows > 13)
        {
            rows = 13;
        }

        if(cols < 2)
        {
            cols = 2;
        }
        else if(cols > 13)
        {
            cols = 13;
        }

        if (useMouse)
        {
            GameObject.Find("Player").gameObject.SetActive(false);
        }
        else
        {
            GameObject.Find("DragPlayer").gameObject.SetActive(false);
        }
        /*SceneManager.sceneLoaded += OnSceneLoaded;
        if (labybool == 1)
        {
            Laby = true;
        }
        else
        {
            Laby = false;
        }*/
        theWalls = new GameObject[cols, rows];
        //Maze initialization loop
        for (int i = 0; i < cols; i++)
        {
            for(int j = 0; j < rows; j++)
            {
                GameObject newWall = Instantiate(wallBlock, new Vector3(i, j, 0), Quaternion.identity);
                newWall.GetComponent<WallScript>().setIndex(i, j);
                theWalls[i,j] = newWall;
            }
        }

        int startCol = Random.Range(0, cols);
        int startRow = Random.Range(0, rows);
        print(startCol + "," + startRow);
        //Generate Maze
        MazeCreation(startCol, startRow);

        //Set player at the initial startCol,startRow node
        starterPos = theWalls[startCol,startRow].transform.position;
        GameObject player = GameObject.FindGameObjectWithTag("Player");
        player.transform.position = starterPos;
        playerMoveScript.walls = theWalls;
        
    }

    private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        labybool = PlayerPrefs.GetInt("laby", 0);
        Laby = (labybool == 1);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    //Returns the 2D array containing the coordinates for wall objects
    public GameObject[,] getWalls()
    {
        return theWalls;
    }

    //Creates a list of unvisited neighbors, later used to randomly determining a neighbor to visit
    private List<GameObject> checkUnvisitedNeighbors(int homeCol, int homeRow)
    {
        //print("hc: " + homeCol + " and hr: " + homeRow);
        List<GameObject> neighbors = new List<GameObject>();
        if(homeCol + 1 < cols)
        {
            TryAddingNeighbor(neighbors, homeCol + 1, homeRow);
        }
        if(homeCol - 1 >= 0)
        {
            //print("1 work");
            TryAddingNeighbor(neighbors, homeCol - 1, homeRow);
        }
        if (homeRow - 1 >= 0)
        {
            //print("2 work");
            TryAddingNeighbor(neighbors, homeCol, homeRow - 1);
        }
        if (homeRow + 1 < rows)
        {
            //print("3 work");
            TryAddingNeighbor(neighbors, homeCol, homeRow + 1);
        }

        //print("List size: " + neighbors.Count);
        return neighbors;
    }

    //Checks to see if a neighbor has been visited yet. If not, add it to the given neighbors list
    private void TryAddingNeighbor(List<GameObject> neighbors, int col, int row)
    {
        bool visited = getVisited(col, row);
        if (!visited)
        {
            neighbors.Add(theWalls[col, row]);
        }
    }

    //Checks to see if a wall node at a given col,row index has been visited
    private bool getVisited(int col, int row)
    {
        return theWalls[col, row].GetComponent<WallScript>().isVisited();
    }

    //Generates the maze given a starting x,y point
    private void MazeCreation(int starterCol, int starterRow)
    {
        Stack<GameObject> walls = new Stack<GameObject>();
        List<GameObject> labyrinth = new List<GameObject>();
        GameObject starterPoint = theWalls[starterCol, starterRow];
        //print("before: " + starterCol + "," + starterRow);
        starterPoint.GetComponent<WallScript>().Visit();
        labyrinth.Add(starterPoint);
        walls.Push(starterPoint);

        //While we have points to check in the walls stack...
        while(walls.Count > 0)
        {
            //Pop the next point and create a list of its unvisited neighbors
            GameObject currentWall = walls.Pop();
            List<GameObject> unvisitedNeighbors = checkUnvisitedNeighbors(currentWall.GetComponent<WallScript>().getCol(), currentWall.GetComponent<WallScript>().getRow());

            //If the unvisitedNeighbors list is not empty, pick a random neighbor to remove walls from and then add it to the stack
            if(unvisitedNeighbors.Count > 0)
            {
                walls.Push(currentWall);
                int randomNeighborIndex = Random.Range(0, unvisitedNeighbors.Count);
                GameObject randomNeighbor = unvisitedNeighbors[randomNeighborIndex];

                //If column index are the same, subtract row indexs to determine which walls need to be removed
                if(randomNeighbor.GetComponent<WallScript>().getCol() == currentWall.GetComponent<WallScript>().getCol())
                {
                    int rowDifference = randomNeighbor.GetComponent<WallScript>().getRow() - currentWall.GetComponent<WallScript>().getRow();
                    //Moving up
                    if (rowDifference == 1)
                    {
                        Destroy(currentWall.transform.Find("Top").gameObject);
                        Destroy(randomNeighbor.transform.Find("Bot").gameObject);
                    }
                    //Moving Down
                    else if (rowDifference == -1)
                    {
                        Destroy(currentWall.transform.Find("Bot").gameObject);
                        Destroy(randomNeighbor.transform.Find("Top").gameObject);
                    }
                }

                //If row index are the same, subtract column indexs to determine which walls need to be removed
                else if (randomNeighbor.GetComponent<WallScript>().getRow() == currentWall.GetComponent<WallScript>().getRow())
                {
                    int colDifference = randomNeighbor.GetComponent<WallScript>().getCol() - currentWall.GetComponent<WallScript>().getCol();
                    //Moving right
                    if (colDifference == 1)
                    {
                        Destroy(currentWall.transform.Find("Right").gameObject);
                        Destroy(randomNeighbor.transform.Find("Left").gameObject);
                    }
                    //Moving left
                    else if (colDifference == -1)
                    {
                        Destroy(currentWall.transform.Find("Left").gameObject);
                        Destroy(randomNeighbor.transform.Find("Right").gameObject);
                    }
                }

                //Mark the neighbor as visited and push to stack
                randomNeighbor.GetComponent<WallScript>().Visit();
                labyrinth.Add(randomNeighbor);
                walls.Push(randomNeighbor);
            }

            else if(unvisitedNeighbors.Count == 0)
            {


                //if Laby is true, do not generate a full maze. Make a labyrinth!
                if (Laby == true)
                {
                    /*foreach (GameObject i in labyrinth)
                    {
                        i.transform.Find("Floor").gameObject.GetComponent<SpriteRenderer>().enabled = true;
                    }*/
                    GameObject player = GameObject.FindGameObjectWithTag("Player");
                    player.transform.position = labyrinth[0].transform.position;
                    GameObject goal = GameObject.FindGameObjectWithTag("Goal");
                    goal.transform.position = labyrinth[labyrinth.Count - 1].transform.position;

                    for (int i = 0; i < cols; i++)
                    {
                        for (int j = 0; j < rows; j++)
                        {
                            if (!labyrinth.Contains(theWalls[i, j]))
                            {
                                theWalls[i, j].gameObject.SetActive(false);
                            }
                        }
                    }
                    return;
                }
                else
                {
                    GameObject player = GameObject.FindGameObjectWithTag("Player");
                    player.transform.position = starterPos;
                    playerMoveScript.walls = theWalls;

                    GameObject goal = GameObject.FindGameObjectWithTag("Goal");
                    while (true)
                    {
                        Vector3 goalPos = new Vector3(Random.Range(0, cols), Random.Range(0, rows), 0);
                        if (goalPos != player.transform.position)
                        {
                            goal.transform.position = goalPos;
                            break;
                        }
                    }
                }
                
            }
            
        }
    }
    /*private IEnumerator visibileGeneration()
    {

    }*/
}


