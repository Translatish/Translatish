import "./static/css/App.css";
import React, { useState, useEffect } from 'react'
import Logo from "./static/media/trans.png";
import Explore from "./Explore";
import './static/css/Navbar.css'
import Home from "./Home";
import AboutUs from "./AboutUs";
import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link,
} from "react-router-dom";

function App() {
  const [activeLink, setAcitveLink] = useState("Home")
  function setlink(link) {
    setAcitveLink(link)
  }
  return (
    <Router>
      <div className="app">
        <nav className="navbar navbar-expand-lg navbar-light bg-transparent">
          <div className="container-fluid">
            <Link className="navbar-brand" to="/">
              <img src={Logo} alt="not found"></img>
            </Link>
            <button
              className="navbar-toggler"
              type="button"
              data-bs-toggle="collapse"
              data-bs-target="#navbarSupportedContent"
              aria-controls="navbarSupportedContent"
              aria-expanded="false"
              aria-label="Toggle navigation"
            >
              <span className="navbar-toggler-icon"></span>
            </button>
            <div className="collapse navbar-collapse" id="navbarSupportedContent">
              <ul className="navbar-nav ms-auto ">
                <li className="nav-item">
                  <Link className={"nav-link" + (activeLink === "Home" ? " active" : "")} onClick={() => setAcitveLink("Home")} aria-current="page" to="/">Home</Link>
                </li>
                <li className="nav-item">
                  <Link className={"nav-link" + (activeLink === "Explore" ? " active" : "")} aria-current="page" onClick={() => setAcitveLink("Explore")} to="/Explore">Explore</Link>
                </li>
                <li className="nav-item">
                  <a className="nav-link" aria-current="page" href="https://github.com/Translatish/Translatish">Github</a>
                </li>
                <li className="nav-item">
                  <Link className={"nav-link" + (activeLink === "AboutUs" ? " active" : "")} aria-current="page" onClick={() => setAcitveLink("AboutUs")} to="/AboutUs">About Us</Link>
                </li>
              </ul>
            </div>
          </div>
        </nav>

        <Switch>
          <Route exact path="/" component={Home} activeLink={setlink} />
          <Route exact path="/Explore" component={Explore} />
          <Route exact path="/AboutUs" component={AboutUs} />

        </Switch>
      </div>
    </Router >
  );
}

export default App;
