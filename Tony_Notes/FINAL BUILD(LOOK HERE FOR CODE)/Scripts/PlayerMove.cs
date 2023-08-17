using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

//THIS IS FOR WASD PLAYER MOVEMENT!
public class PlayerMove : MonoBehaviour
{
    public GameObject[,] walls;
    private GameObject goal;
    public GameObject MazeManager;
    // Start is called before the first frame update
    void Start()
    {
        walls = MazeManager.GetComponent<MazeGenerator>().getWalls();
        goal = GameObject.FindGameObjectWithTag("Goal");
        //print(walls.Length);
    }

    // Update is called once per frame
    void Update()
    {
        //Input detection for player movement
        if (Input.GetKeyDown(KeyCode.W))
        {
            attemptMove(new Vector3(0, 1, 0),"Bot");
        }
        else if (Input.GetKeyDown(KeyCode.S))
        {
            attemptMove(new Vector3(0, -1, 0),"Top");
        }
        else if (Input.GetKeyDown(KeyCode.A))
        {
            attemptMove(new Vector3(-1, 0, 0),"Right");
        }
        else if (Input.GetKeyDown(KeyCode.D))
        {
            attemptMove(new Vector3(1, 0, 0),"Left");
        }
    }

    //attempts to move in the given direction
    private void attemptMove(Vector3 direction, string WallDirection)
    {
        Vector3 newPos = gameObject.transform.position;
        Vector3 desiredPos = newPos + direction;
        if (canMove(desiredPos,WallDirection))
        {
            gameObject.transform.position = desiredPos;
        }
    }

    //checks to see if the requested move is valid, or if the player hits a goal
    private bool canMove(Vector3 desiredPos, string WallDirection)
    {
        GameObject currentWall = walls[(int)desiredPos.x, (int)desiredPos.y];
        if (goal.transform.position == desiredPos)
        {
            SceneManager.LoadScene("Win");
        }
        else if(currentWall != null && currentWall.transform.Find(WallDirection) != null)
        {
            return false;
        }
        /*foreach(GameObject i in walls)
        {
            if (i != null)
            {
                if (i.transform.position == desiredPos &)
                {
                    return false;
                }
            }
            
        }*/
        return true;
    }
}
