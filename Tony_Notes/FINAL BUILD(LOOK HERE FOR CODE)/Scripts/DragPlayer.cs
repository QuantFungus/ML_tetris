using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
//using UnityEngine.InputSystem;


public class DragPlayer : MonoBehaviour
{
    private float _mouseSensitivity = 5f;
    public bool controlling = false;
    public bool canclick = true;
    private GameObject MazeManager;

    public Vector2 spawn;

    private Vector2 mouseVelocity = Vector2.zero;
    // Start is called before the first frame update
    void Start()
    {
        MazeManager = GameObject.Find("MazeManager");
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0) && canclick && !controlling) //&& mouseOverPlayer())
        {
            controlling = true;
            StartCoroutine(ClickCD());
        }

        if (controlling && canclick && Input.GetMouseButtonDown(0))
        {
            controlling = false;
            StartCoroutine(ClickCD());
            //Mouse.current.WarpCursorPosition(transform.position);
        }

        mouseVelocity = new Vector2(Input.GetAxis("Mouse X"), Input.GetAxis("Mouse Y"));

        if (controlling == true)
        {
           // print("aaaaa");
            //gameObject.transform.position = Input.mousePosition;
            gameObject.GetComponent<Rigidbody2D>().AddForce(mouseVelocity * _mouseSensitivity, ForceMode2D.Impulse);
            print(mouseVelocity + " " + _mouseSensitivity);
            Cursor.visible = false;
        }
        else
        {
            Cursor.visible = true;
            Cursor.lockState = CursorLockMode.None;
        }
    }

    //checks to see if the mouse is hovering over the player
    private bool mouseOverPlayer()
    {
        //print("aatempt");
        PointerEventData pointerData = new PointerEventData(EventSystem.current);
        pointerData.position = Input.mousePosition;

        List<RaycastResult> results = new List<RaycastResult>();
        EventSystem.current.RaycastAll(pointerData, results);
        for (int i = 0; i < results.Count; i++)
        {
            print(results[i].gameObject.tag);
            if (results[i].gameObject.tag != "Player")
            {
                
                results.RemoveAt(i);
                i--;
            }
        }
        return results.Count > 0;
    }

    //starts a cooldown so the player can't click immediately after clicking
    private IEnumerator ClickCD()
    {
        canclick = false;

        yield return new WaitForSeconds(0.5f);

        canclick = true;
    }

    //checks if the player runs into anything
    private void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.gameObject.tag == "Goal")
        {
            print("YOOOOOOOOOOOOOOOOOOOO");

            controlling = false;
            Cursor.visible = true;
            Cursor.lockState = CursorLockMode.None;
            SceneManager.LoadScene("Win");
            //Mouse.current.WarpCursorPosition(transform.position);
        }
        else if (collision.gameObject.tag == "Wall")
        {
            controlling = false;




            /*if (collision.gameObject.name == "Right")
            {
                gameObject.GetComponent<Rigidbody2D>().AddForce(new Vector2(-1000, 0));
            }
            else if(collision.gameObject.name == "Left")
            {
                gameObject.GetComponent<Rigidbody2D>().AddForce(new Vector2(1000, 0));
            }
            else if (collision.gameObject.name == "Top")
            {
                gameObject.GetComponent<Rigidbody2D>().AddForce(new Vector2(0, -1000));
            }
            else if (collision.gameObject.name == "Bot")
            {
                gameObject.GetComponent<Rigidbody2D>().AddForce(new Vector2(0, 1000));
            }*/
        }
    }
}

