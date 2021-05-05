import React, { Component } from 'react'
import PropTypes from 'prop-types'
import rb from './static/media/rahul_chocha.jpg'
import mt from './static/media/mayank_tank.jpeg'
import yp from './static/media/yash_patel.jpeg'
import dp from './static/media/dipika_pawar.jpeg'
import ap from './static/media/anshi_patwari.jpeg'
import './static/css/AboutUs.css'

export default class AboutUs extends Component {

    render() {
        return (
            <div className="About_us">
                <div className="aboutUs__con">
                    <div className="about__text">
                        Meet the Team
                    </div>

                    <div className="card__con">
                        <a className="card__link" href="https://github.com/mrchocha/" >
                            <div className="card mb-3">
                                <div className="card__img">
                                    <img src={rb} className="card-img-top" alt="..." />
                                </div>
                                <div className="card-body">
                                    <h5 className="card-title">Rahul Chocha</h5>
                                    <p className="card-text">School of Engineering and Applied Science</p>
                                    <p className="card-text"><small className="text-muted">B.Tech 3rd year</small></p>
                                </div>
                            </div>
                        </a>
                        <a className="card__link" href="https://github.com/Mayankkumartank" >
                            <div className="card mb-3">
                                <div className="card__img">
                                    <img src={mt} className="card-img-top" alt="..." />
                                </div>
                                <div className="card-body">
                                    <h5 className="card-title">Mayankkumar Tank</h5>
                                    <p className="card-text">School of Engineering and Applied Science</p>
                                    <p className="card-text"><small className="text-muted">B.Tech 3rd year</small></p>
                                </div>
                            </div>
                        </a>
                        <a className="card__link" href="https://github.com/DipikaPawar12" >
                            <div className="card mb-3">
                                <div className="card__img">
                                    <img src={dp} className="card-img-top" alt="..." />
                                </div>
                                <div className="card-body">
                                    <h5 className="card-title">Dipika Pawar</h5>
                                    <p className="card-text">School of Engineering and Applied Science</p>
                                    <p className="card-text"><small className="text-muted">B.Tech 3rd year</small></p>
                                </div>
                            </div>
                        </a>

                    </div>
                    <div className="card__con">
                        <a className="card__link" href="https://github.com/yash982000" >
                            <div className="card mb-3">
                                <div className="card__img">
                                    <img src={yp} className="card-img-top" alt="..." />
                                </div>
                                <div className="card-body">
                                    <h5 className="card-title">Yash Patel</h5>
                                    <p className="card-text">School of Engineering and Applied Science</p>
                                    <p className="card-text"><small className="text-muted">B.Tech 3rd year</small></p>
                                </div>
                            </div>
                        </a>
                        <a className="card__link" href="https://github.com/aanshi18" >
                            <div className="card mb-3">
                                <div className="card__img">
                                    <img src={ap} className="card-img-top" alt="..." />
                                </div>
                                <div className="card-body">
                                    <h5 className="card-title">Anshi Patwari</h5>
                                    <p className="card-text">School of Engineering and Applied Science</p>
                                    <p className="card-text"><small className="text-muted">B.Tech 3rd year</small></p>
                                </div>
                            </div>
                        </a>
                    </div>
                </div>
            </div>
        )
    }
}
