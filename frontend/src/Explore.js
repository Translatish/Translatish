import React, { useState, useEffect } from 'react'
import ReactPlayer from 'react-player'
import './static/css/Explore.css'
import FileUpload from './File_upload'
import AOS from 'aos';

function Explore() {
    const [videoFilePath, setVideoFileURL] = useState(null);
    const [HindiText, setHindiText] = useState("");
    const [EnglishText, setEnglishText] = useState("");

    useEffect(() => {
        AOS.init();
    }, []);

    function getData(files) {
        const formData = new FormData()
        formData.append('myFile', files)

        fetch("http://127.0.0.1:8000/", {
            method: 'POST',
            body: formData
        })
            .then(res => res.json())
            .then(res => {
                setHindiText(res.hindi);
                setEnglishText(res.english);
            })
    }
    return (
        <div className="Explore">
            <FileUpload setVideoFileURL={setVideoFileURL} getdata={getData} data-aos="fade-down" />
            <div className="video_cont__main">
                {
                    videoFilePath ?
                        <div className="Video__con">
                            <div div className="video__text"> Uploaded Video</div>
                            <div className="video__player__con" data-aos="zoom-in" data-aos-easing="ease-out-cubic" data-aos-duration="1000">
                                <ReactPlayer className="video__player"
                                    url={videoFilePath} controls={true} />
                            </div>
                        </div>
                        : ""
                }
            </div >
            <div className="row row__textData" data-aos="fade-down">
                <div className="col-sm col__left" >
                    {
                        EnglishText ? EnglishText : "div1"
                    }
                </div>
                <div className="col-sm col__right" >
                    {
                        HindiText ? HindiText : "div2"
                    }
                </div>
            </div>
        </div >
    )
}

export default Explore
