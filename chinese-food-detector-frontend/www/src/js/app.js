import React, { Component } from 'react';
import { render } from 'react-dom';
import { BrowserRouter as Router, Route, Link } from 'react-router-dom'
import { Nav,Navbar, NavItem } from 'react-bootstrap';

// require('react-fastclick');

import HomePage from './pages/home';
import ResultPage from './pages/result';


render(
    <Router>
      <div>
        <Navbar inverse collapseOnSelect>
          <Navbar.Header>
            <Navbar.Brand>
              <div>Chinese Food Detector</div>
            </Navbar.Brand>
            <Navbar.Toggle />
          </Navbar.Header>
          <Navbar.Collapse>
            <Nav>
              <NavItem eventKey={1} ><Link to="/">Home</Link></NavItem>
              <NavItem eventKey={2} ><Link to="/result">Result</Link></NavItem>
            </Nav>
          </Navbar.Collapse>
        </Navbar>

        <Route exact path="/" component={HomePage}/>
        <Route path="/result" component={ResultPage}/>
      </div>
    </Router>,
    document.getElementById('app')
);
