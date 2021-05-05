import React, { useEffect } from 'react'
import home_img from './static/media/home.jpg'
import './static/css/Home.css'
import AOS from 'aos';
import { NavLink } from 'react-router-dom'
function Home(props) {
    useEffect(() => {
        AOS.init();
    }, []);
    return (
        <div className="Home">
            <div className="row" >
                <div className="col-sm col__left" data-aos="fade-left" >
                    <div className="Home__text">
                        <span className="welcome">Welcome to </span><span className="main__text">Translatish</span>
                    </div>
                    <div className="Home__subText">
                        we are Translatish convert English video to hindi in just one click...
                    </div>
                    <NavLink to="/Explore" >
                        <div className="button_con">
                            <button className="Explore__button btn btn-group btn-group-lg" role="group">
                                Explore
                        </button>
                        </div>
                    </NavLink>


                </div>
                <div className="col-sm col__right">
                    <img data-aos="fade-right" src={home_img} alt="Not found"></img>
                </div>
            </div>

        </div>
    )
}

export default Home
