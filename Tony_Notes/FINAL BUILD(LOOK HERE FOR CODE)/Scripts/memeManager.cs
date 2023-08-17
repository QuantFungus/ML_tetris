using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class memeManager : MonoBehaviour
{
    public List<GameObject> memeList = new List<GameObject>();
    // Start is called before the first frame update
    void Start()
    {
        GameObject canvas = GameObject.Find("Canvas");
        int randomNumber = Random.Range(0, memeList.Count);
        GameObject newMeme = Instantiate(memeList[randomNumber]);
        newMeme.transform.SetParent(canvas.transform);
        newMeme.transform.position = new Vector2(650, 250);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
